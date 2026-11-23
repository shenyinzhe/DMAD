# KITScenes Multimodal: open-loop inference

Zero-shot open-loop inference of the nuScenes-trained DMAD checkpoint on the
Multimodal split of KITScenes (483 E2E keyframes in `val` and
`overlap_train_val`). The adapter maps the six KITScenes ring cameras, ego
state and driving command onto the nuScenes conventions DMAD was trained with.
UniAD runs with the same adapter after a one-step setup, see
[section 5](#5-uniad).

## 1. Environment, checkpoint, motion anchors

- Environment: follow [INSTALL.md](INSTALL.md) (torch 1.9.1, mmcv-full 1.4.0,
  mmdet3d 0.17.1), or build the image in `docker/`.
- Checkpoint: download
  [dmad_stage2.pth](https://github.com/shenyinzhe/DMAD/releases/download/v1.0/dmad_stage2.pth)
  to `ckpts/dmad_stage2.pth`.
- Motion anchors: `data/others/motion_anchor_infos_mode6.pkl`, as described in
  [DATA_PREP.md](DATA_PREP.md). The nuScenes data and info files are not needed
  for this inference.

## 2. Dataset definition

You need the KITScenes Multimodal release data and its E2E dataset definition
(JSON). Each keyframe entry stores an absolute `data_root`
(`<release>/data/<split>`). Point it at your local copy:

```shell
python tools/kitscenes/prepare_dataset_json.py \
    --src /path/to/multimodal_e2e_dataset.json \
    --data-root /path/to/kitscenes_release/data \
    --out data/kitscenes/multimodal_e2e_dataset.json
```

Only `data_root` is changed. The script also checks that the three queue frames
of every keyframe (offsets -10, -5, 0 at 10 Hz) have a pose and six images.

## 3. Inference

```shell
python tools/kitscenes/inference.py \
    --config projects/configs/stage2_e2e/dmad_kitscenes_multimodal.py \
    --checkpoint ckpts/dmad_stage2.pth \
    --out-dir work_dirs/dmad_kitscenes_multimodal
```

- Run from the repo root with the repo on `PYTHONPATH`.
- Multi-GPU: set `CUDA_VISIBLE_DEVICES=0,1,2,3`. The script starts one worker
  per GPU and shards by clip, so each temporal queue stays on one worker.
- `--limit N` runs only the first N samples (smoke test). `--ann-file` overrides
  the dataset JSON of the config.
- Output: `pred_<uuid>_kf<keyframe>.npy` per keyframe, the planner output
  `sdc_traj_all` of shape (1, 6, 2): six 0.5 s steps (3 s) in nuScenes lidar
  axes (x right, y forward).

Input conventions (`projects/mmdet3d_plugin/datasets/kitscenes_multimodal_dataset.py`):

- **Cameras**: every ring image is resampled by a homography into the matching
  virtual nuScenes camera (1600x900, median nuScenes intrinsics and camera
  orientation). KITScenes ring cameras are much wider (~87 deg versus ~64 deg),
  so a plain resize would show objects at ~0.6x their trained size.
- **Ego state (CAN bus)**: causal backward differences at 10 Hz for velocity,
  acceleration and yaw rate, in the ego frame. Only past poses are read.
- **Command**: the 3 s label `high_level_instruction_3s`, matching the 3 s
  planning horizon. Turn right -> 0, turn left / u-turn -> 1, everything
  else -> 2 (as in nuScenes).
- **Queue**: frames at -1.0 s, -0.5 s and 0 s; the keyframe is the current frame.

## 4. Export

```shell
python tools/kitscenes/export_jsonl.py \
    --pred-dir work_dirs/dmad_kitscenes_multimodal \
    --out work_dirs/predictions_dmad_kitscenes_multimodal.jsonl \
    --gt /path/to/gt_future_50pts.jsonl   # optional
```

Each line is `{"scenario_id", "keyframe_id", "prediction_frequency_hz": 2.0,
"future_trajectory": [[x_forward, y_left] x 6]}`. With `--gt`, the script also
prints L2 at 1/2/3 s and ADE against the 10 Hz ground truth. This is only a
local sanity check; the official KITScenes metric is computed by the benchmark.

Optional overlay of the predicted trajectory on the front camera:

```shell
python tools/kitscenes/visualize.py \
    --traj-dir work_dirs/dmad_kitscenes_multimodal \
    --e2e-json data/kitscenes/multimodal_e2e_dataset.json \
    --out-dir work_dirs/dmad_kitscenes_multimodal_vis
```

## 5. UniAD

The same adapter and tools run the official UniAD checkpoint. Install them into
an official UniAD v1 checkout (branch `main`; the default branch `v2.0` is a
different code base). From the DMAD repository root:

```shell
git clone https://github.com/OpenDriveLab/UniAD.git
git -C UniAD checkout main
bash tools/kitscenes/setup_uniad.sh /path/to/UniAD
```

The script copies `kitscenes_multimodal_dataset.py`, the config
`uniad_kitscenes_multimodal.py` and `tools/kitscenes/` into UniAD and applies
`tools/kitscenes/uniad/uniad_kitscenes.patch` (dataset registration and
test-time guards for inputs without GT). The environment is the same as for
DMAD.

Then, in the UniAD checkout, download
[uniad_base_e2e.pth](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth)
to `ckpts/`, place the
[motion anchors](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl)
in `data/others/`, prepare the dataset JSON as in section 2, and run:

```shell
python tools/kitscenes/inference.py \
    --config projects/configs/stage2_e2e/uniad_kitscenes_multimodal.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --out-dir work_dirs/uniad_kitscenes_multimodal
```

Export and visualization work as in section 4.
