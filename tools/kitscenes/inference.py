"""Run DMAD inference on the KITScenes Multimodal keyframes and dump the
predicted future trajectories.

The dataset is described by the KITScenes Multimodal E2E dataset JSON
(see docs/KITSCENES.md).

Output: one file per keyframe in `work_dirs/kitscenes_preds/`, named
`pred_<uuid>_kf<keyframe>.npy`. Each file is the model's
sdc_traj_all (shape (1, 6, 2) — 6 future steps × (x, y) at 0.5 s spacing).

Usage (from the repo root, inside apptainer):
    python tools/kitscenes/inference.py

Multi-GPU usage: set CUDA_VISIBLE_DEVICES to a comma-separated list and the
script auto-forks one worker process per device, each owning a disjoint
subset of clips:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/kitscenes/inference.py

DMAD's `prev_frame_info` is per-process so cross-shard temporal state never
leaks; each shard handles whole clips (queue stays contiguous) so prev_bev
accumulates correctly within a clip.
"""
import os
import sys
import argparse
import subprocess


def output_filename_for_info(info):
    """File name for one keyframe — uniquely identifies it."""
    uuid = info['uuid']
    kf = int(info['keyframe'])
    return f'pred_{uuid}_kf{kf:010d}.npy'


def resolve_shards_from_env(cuda_visible):
    """Parse CUDA_VISIBLE_DEVICES into a list of GPU ids.

    Returns [] if unset/empty (caller should run inline on the default device).
    """
    if cuda_visible is None:
        return []
    parts = [p.strip() for p in str(cuda_visible).split(',')]
    return [p for p in parts if p]


def shard_data_infos(infos, shard, num_shards):
    """Split data_infos by clip so each shard owns whole queues.

    `is_first_in_scene=True` marks the start of a clip's frame queue; we
    walk the list, group consecutive frames into clips, and keep clips
    where (clip_idx % num_shards) == shard.
    """
    if num_shards <= 1:
        return list(infos)
    groups = []
    current = []
    for info in infos:
        if info.get('is_first_in_scene', False) and current:
            groups.append(current)
            current = []
        current.append(info)
    if current:
        groups.append(current)
    out = []
    for i, g in enumerate(groups):
        if i % num_shards == shard:
            out.extend(g)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',
                   default='projects/configs/stage2_e2e/dmad_kitscenes_multimodal.py')
    p.add_argument('--checkpoint', default='ckpts/dmad_stage2.pth')
    p.add_argument('--out-dir', default='work_dirs/kitscenes_preds')
    p.add_argument('--ann-file', default=None,
                   help='Override the dataset JSON (data.test.ann_file) of the config.')
    p.add_argument('--limit', type=int, default=0,
                   help='Limit to first N samples (0 = all). For smoke tests.')
    p.add_argument('--shard', type=int, default=None,
                   help='[internal] Shard index for this worker.')
    p.add_argument('--num-shards', type=int, default=None,
                   help='[internal] Total number of shard workers.')
    return p.parse_args()


def _run_one_shard(args):
    """Actual inference loop for a single shard / single GPU."""
    # Heavy imports deferred so the launcher path doesn't pay for them.
    import numpy as np
    import torch
    import mmcv
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    import projects.mmdet3d_plugin  # noqa: F401  (registers KITScenesMultimodalDataset)

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file
    dataset = build_dataset(cfg.data.test)

    shard = args.shard if args.shard is not None else 0
    num_shards = args.num_shards if args.num_shards is not None else 1
    if num_shards > 1:
        kept = shard_data_infos(dataset.data_infos, shard, num_shards)
        dropped = len(dataset.data_infos) - len(kept)
        dataset.data_infos = kept
        print(f'[shard {shard}/{num_shards}] kept {len(kept)} samples '
              f'(dropped {dropped} owned by other shards)')

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    tag = f'shard {shard}/{num_shards}' if num_shards > 1 else 'single'
    print(f'[{tag}] running inference on {len(dataset)} samples '
          f'({len(dataset) // max(1, dataset.queue_length)} clips)')
    bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        data = scatter(data, [0])[0]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        traj = _extract_traj(result)

        info = dataset.data_infos[i]
        if info.get('is_current', False):
            if traj is None:
                print(f'\n[warn] no sdc_traj_all for {info["scene_token"]}')
                traj = np.zeros((1, 6, 2), dtype=np.float32)
            out_path = os.path.join(args.out_dir, output_filename_for_info(info))
            try:
                np.save(out_path, traj)
            except Exception as e:
                print(f'\n[warn] failed to save {out_path}: {e}')
        bar.update()
    print(f'\n[{tag}] done.')


def _extract_traj(result):
    import torch
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        result = result[0]
    if 'sdc_traj_all' in result:
        traj = result['sdc_traj_all']
    elif 'planning' in result and 'result_planning' in result['planning'] \
            and 'sdc_traj_all' in result['planning']['result_planning']:
        traj = result['planning']['result_planning']['sdc_traj_all']
    else:
        return None
    if isinstance(traj, torch.Tensor):
        traj = traj.cpu().numpy()
    return traj


def _spawn_shards(args, gpu_ids):
    """Fork one subprocess per GPU listed in CUDA_VISIBLE_DEVICES."""
    procs = []
    n = len(gpu_ids)
    for i, gpu in enumerate(gpu_ids):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--config', args.config,
            '--checkpoint', args.checkpoint,
            '--out-dir', args.out_dir,
            '--shard', str(i),
            '--num-shards', str(n),
        ]
        if args.limit:
            cmd += ['--limit', str(args.limit)]
        if args.ann_file:
            cmd += ['--ann-file', args.ann_file]
        print(f'[launcher] spawning shard {i}/{n} on GPU {gpu}')
        procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    for p in procs:
        ret = p.wait()
        if ret != 0:
            rc = ret
            print(f'[launcher] shard pid={p.pid} exited with code {ret}')
    return rc


def main():
    args = parse_args()

    # Worker invocation (launched by ourselves): just run the shard.
    if args.shard is not None and args.num_shards is not None:
        _run_one_shard(args)
        return 0

    # Top-level invocation: decide single vs multi based on CUDA_VISIBLE_DEVICES.
    gpu_ids = resolve_shards_from_env(os.environ.get('CUDA_VISIBLE_DEVICES'))
    if len(gpu_ids) > 1:
        return _spawn_shards(args, gpu_ids)

    # Single GPU (or unset): run inline, no fork.
    args.shard = 0
    args.num_shards = 1
    _run_one_shard(args)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
