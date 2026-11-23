"""Convert the `tools/kitscenes/inference.py` dumps to the KITScenes prediction JSONL.

Input: `<pred-dir>/pred_<uuid>_kf<keyframe>.npy`, each the planner output
`sdc_traj_all` of shape (1, 6, 2) in nuScenes lidar axes (x right, y forward),
6 steps at 0.5 s.

Output, one line per keyframe:
    {"scenario_id": <uuid>, "keyframe_id": "<kf>", "prediction_frequency_hz": 2.0,
     "future_trajectory": [[x_forward, y_left] x 6]}

With `--gt gt_future_50pts.jsonl` (10 Hz ground truth) it also prints a local
L2 sanity check at 1/2/3 s and ADE over the six points. This is not the official
KITScenes metric, which is computed by the benchmark.

    python tools/kitscenes/export_jsonl.py --pred-dir work_dirs/dmad_kitscenes_multimodal \
        --out work_dirs/predictions_dmad_kitscenes_multimodal.jsonl [--gt /path/to/gt_future_50pts.jsonl]
"""
import argparse
import glob
import json
import os

import numpy as np


def load_preds(pred_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, 'pred_*_kf*.npy'))):
        stem = os.path.basename(f)[len('pred_'):-len('.npy')]
        uuid, kf = stem.rsplit('_kf', 1)
        arr = np.load(f)
        assert arr.shape == (1, 6, 2), (f, arr.shape)
        xy = arr[0].astype(np.float64)
        out[(uuid, str(int(kf)))] = np.stack([xy[:, 1], -xy[:, 0]], 1)  # forward, left
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gt', default=None, help='optional gt_future_50pts.jsonl for local L2')
    a = p.parse_args()

    preds = load_preds(a.pred_dir)
    if not preds:
        raise SystemExit(f'no pred_*_kf*.npy files in {a.pred_dir}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        for key in sorted(preds):
            fh.write(json.dumps({
                'scenario_id': key[0], 'keyframe_id': key[1],
                'prediction_frequency_hz': 2.0,
                'future_trajectory': [[float(x), float(y)] for x, y in preds[key]],
            }) + '\n')
    print(f'wrote {len(preds)} keyframes to {a.out}')

    if a.gt:
        gt = {}
        for line in open(a.gt):
            r = json.loads(line)
            assert r['prediction_frequency_hz'] == 10.0
            gt[(r['scenario_id'], str(r['keyframe_id']))] = np.asarray(r['future_trajectory'])[:, :2]
        keys = sorted(set(preds) & set(gt))
        print(f'{len(keys)} keyframes with GT, {len(set(gt) - set(preds))} GT keyframes without prediction')
        err = np.stack([np.linalg.norm(preds[k] - gt[k][4::5][:6], axis=1) for k in keys])
        print('L2@1s {:.3f}  L2@2s {:.3f}  L2@3s {:.3f}  ADE {:.3f}  (m, local sanity check)'.format(
            err[:, 1].mean(), err[:, 3].mean(), err[:, 5].mean(), err.mean()))


if __name__ == '__main__':
    main()
