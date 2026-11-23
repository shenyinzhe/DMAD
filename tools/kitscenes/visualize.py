"""Overlay DMAD-predicted future trajectories from `tools/kitscenes/inference.py` onto the
front camera image of each KITScenes keyframe.

Reads:
  - work_dirs/kitscenes_preds/pred_<uuid>_kf<keyframe>.npy
  - data/kitscenes/multimodal_e2e_dataset.json (see docs/KITSCENES.md)

Writes:
  - work_dirs/kitscenes_preds_vis/<uuid>_kf<keyframe>.jpg
"""
import os
import sys
import glob
import json
import argparse

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.datasets.kitscenes_multimodal_dataset import (  # noqa: E402
    _orthogonalize, HLI_TO_COMMAND,
)


COMMAND_LABEL = {0: 'Right', 1: 'Left', 2: 'Straight'}


def map_command(raw_text):
    """Same logic as KITScenesMultimodalDataset.get_data_info: lower+strip then dict lookup,
    with default 2 (Straight). Returns (code, label_str)."""
    key = str(raw_text or '').strip().lower()
    code = HLI_TO_COMMAND.get(key, 2)
    return code, COMMAND_LABEL.get(code, str(code))


def lidar2img_for_front(calib, img_w, img_h, native_res=(3504, 2272)):
    """Build NuScenes-lidar -> front-camera-image projection matrix.

    `calib` is an e2e_dataset.json `calib` block (alias-keyed, each entry
    holds `K`, `R`, `t`, optionally `resolution`).
    """
    cam = calib['fv_m']
    K = np.asarray(cam['K'], dtype=np.float64).copy()
    R_cb = _orthogonalize(np.asarray(cam['R'], dtype=np.float64))
    t_cb = np.asarray(cam['t'], dtype=np.float64)
    res = cam.get('resolution') or {'width': native_res[0], 'height': native_res[1]}

    scale_x = img_w / float(res['width'])
    scale_y = img_h / float(res['height'])
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y

    T_cam_to_base = np.eye(4, dtype=np.float64)
    T_cam_to_base[:3, :3] = R_cb
    T_cam_to_base[:3, 3] = t_cb
    T_base_to_cam = np.linalg.inv(T_cam_to_base)

    nusc2base = np.array([
        [0.0,  1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0,  0.0, 0.0, 1.0],
    ], dtype=np.float64)

    lidar2cam = T_base_to_cam @ nusc2base

    lidar2img = np.eye(4, dtype=np.float64)
    lidar2img[:3, :3] = K
    lidar2img = lidar2img @ lidar2cam
    return lidar2img, lidar2cam


def project_traj(traj_xy, lidar2img, ground_z=0.0):
    """Project (N, 2) DMAD sdc_traj_all points onto the image.

    Empirical convention (verified against val-split HLI labels):
      traj[..., 0] = lateral, RIGHT positive
      traj[..., 1] = forward
    The dataset's lidar2img treats nusc-lidar as (X-right, Y-fwd, Z-up), so
    we feed col0 directly into X and col1 into Y. (The previous DMAD
    long-tail visualizer negated col0 — that was a bug; on data where the
    car mostly drives straight col0~=0 hides it.)
    """
    N = traj_xy.shape[0]
    X_nusc = traj_xy[:, 0]   # right
    Y_nusc = traj_xy[:, 1]   # forward
    Z_nusc = np.full(N, ground_z, dtype=np.float64)
    ones = np.ones(N, dtype=np.float64)
    pts = np.stack([X_nusc, Y_nusc, Z_nusc, ones], axis=1)  # (N, 4)
    proj = pts @ lidar2img.T
    eps = 1e-5
    u = proj[:, 0] / (proj[:, 2] + eps)
    v = proj[:, 1] / (proj[:, 2] + eps)
    z = proj[:, 2]
    uv = np.stack([u, v], axis=1)
    return uv, z > 0.1


def _parse_filename(basename):
    """`pred_<uuid>_kf<keyframe>.npy` -> (uuid, keyframe_int) or None."""
    if not (basename.startswith('pred_') and basename.endswith('.npy')):
        return None
    stem = basename[len('pred_'):-len('.npy')]
    if '_kf' not in stem:
        return None
    uuid, kf_str = stem.rsplit('_kf', 1)
    try:
        return uuid, int(kf_str)
    except ValueError:
        return None


def _build_e2e_lookup(e2e_path):
    """Index e2e_dataset.json by (uuid, keyframe int) -> entry dict."""
    with open(e2e_path) as f:
        data = json.load(f)
    lookup = {}
    for uuid, uuid_entry in data.items():
        if uuid == 'metadata' or not isinstance(uuid_entry, dict):
            continue
        for kf_str, entry in uuid_entry.items():
            try:
                lookup[(uuid, int(kf_str))] = entry
            except ValueError:
                continue
    return lookup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--traj-dir', default='work_dirs/kitscenes_preds')
    p.add_argument('--out-dir', default='work_dirs/kitscenes_preds_vis')
    p.add_argument(
        '--e2e-json',
        default='data/kitscenes/multimodal_e2e_dataset.json',
        help='KITScenes e2e_dataset JSON (see docs/KITSCENES.md)',
    )
    p.add_argument('--ground-z', type=float, default=-1.5,
                   help='Ground plane height in nusc-lidar Z (default approx -1.5m)')
    p.add_argument('--limit', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isfile(args.e2e_json):
        raise SystemExit(f'e2e_dataset.json not found at {args.e2e_json}; '
                         'see docs/KITSCENES.md.')
    lookup = _build_e2e_lookup(args.e2e_json)

    npy_files = sorted(glob.glob(os.path.join(args.traj_dir, 'pred_*.npy')))
    print(f'Found {len(npy_files)} trajectory files.')

    written = 0
    skipped = 0

    for k, npy_path in enumerate(npy_files):
        if args.limit and k >= args.limit:
            break
        basename = os.path.basename(npy_path)
        parsed = _parse_filename(basename)
        if parsed is None:
            print(f'[skip] cannot parse filename {basename}')
            skipped += 1
            continue
        uuid, kf = parsed

        if (uuid, kf) not in lookup:
            print(f'[skip] no e2e entry for {uuid} kf={kf}')
            skipped += 1
            continue
        entry = lookup[(uuid, kf)]

        data_root = entry['data_root']
        rel = entry.get('images_past', {}).get(str(kf), {}).get('fv_m')
        if not rel:
            print(f'[skip] {uuid} kf={kf}: no fv_m image at keyframe')
            skipped += 1
            continue
        img_path = os.path.join(data_root, rel)
        if not os.path.exists(img_path):
            print(f'[skip] {uuid} kf={kf}: missing {img_path}')
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        lidar2img, _ = lidar2img_for_front(entry['calib'], w, h)

        traj = np.load(npy_path)
        if traj.ndim == 3:
            traj = traj[0]  # (6, 2)

        uv, mask = project_traj(traj, lidar2img, ground_z=args.ground_z)

        raw_cmd = entry.get('high_level_instruction_3s', '')
        code, label = map_command(raw_cmd)
        cv2.putText(img, f'Raw HLI: {raw_cmd or "<empty>"}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5,
                    cv2.LINE_AA)
        cv2.putText(img, f'DMAD cmd: {code} ({label})', (50, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 5,
                    cv2.LINE_AA)

        for i in range(len(uv)):
            if not mask[i]:
                continue
            pt = (int(uv[i, 0]), int(uv[i, 1]))
            if -2000 <= pt[0] < w + 2000 and -2000 <= pt[1] < h + 2000:
                cv2.circle(img, pt, 15, (0, 0, 255), -1)
                if i > 0 and mask[i - 1]:
                    prev = (int(uv[i - 1, 0]), int(uv[i - 1, 1]))
                    cv2.line(img, prev, pt, (0, 255, 0), 6)

        out_path = os.path.join(args.out_dir, f'{uuid}_kf{kf:010d}.jpg')
        cv2.imwrite(out_path, img)
        written += 1
        if written % 25 == 0:
            print(f'  wrote {written} images...')

    print(f'Done. Wrote {written}, skipped {skipped} -> {args.out_dir}')


if __name__ == '__main__':
    main()
