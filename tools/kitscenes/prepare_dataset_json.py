"""Point a KITScenes E2E dataset definition at a local copy of the release data.

Every keyframe entry of the KITScenes Multimodal E2E dataset JSON stores an absolute `data_root`
of the form `<release>/data/<split>`. This script rewrites only that field to
`<--data-root>/<split>` (the last path component is kept) and leaves every other
field unchanged. It also checks that the three queue frames read by UniAD/DMAD
(offsets -10, -5, 0 at 10 Hz) have a pose and all six images on disk.

    python tools/kitscenes/prepare_dataset_json.py \
        --src /path/to/multimodal_e2e_dataset.json \
        --data-root /path/to/kitscenes_release/data \
        --out data/kitscenes/multimodal_e2e_dataset.json
"""
import argparse
import collections
import hashlib
import json
import os


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='original KITScenes Multimodal E2E dataset JSON')
    p.add_argument('--data-root', required=True,
                   help='local release data directory that contains val/ and overlap_train_val/')
    p.add_argument('--out', default='data/kitscenes/multimodal_e2e_dataset.json')
    p.add_argument('--no-check', action='store_true', help='skip the image existence check')
    a = p.parse_args()

    raw = open(a.src, 'rb').read()
    d = json.loads(raw)
    stats = collections.Counter()
    missing = []
    for uid, kfs in d.items():
        if uid == 'metadata':
            continue
        for kf, e in kfs.items():
            split_dir = os.path.basename(os.path.normpath(e['data_root']))
            e['data_root'] = os.path.join(a.data_root, split_dir)
            stats['keyframes'] += 1
            k = int(kf)
            for fi in (k - 10, k - 5, k):
                if str(fi) not in e['traj_past']:
                    stats['queue_pose_missing'] += 1
                imgs = e['images_past'].get(str(fi), {})
                if len(imgs) != 6:
                    stats['queue_imgset_incomplete'] += 1
                if a.no_check:
                    continue
                for rel in imgs.values():
                    stats['images_checked'] += 1
                    path = os.path.join(e['data_root'], rel)
                    if not os.path.isfile(path):
                        missing.append(path)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(d, f)
    report = dict(src=a.src, src_sha256=hashlib.sha256(raw).hexdigest(), out=a.out,
                  data_root=a.data_root, stats=dict(stats),
                  n_missing_images=len(missing), missing_images=missing[:20])
    print(json.dumps(report, indent=1))
    if missing or stats['queue_pose_missing'] or stats['queue_imgset_incomplete']:
        raise SystemExit('incomplete inputs, see the report above')


if __name__ == '__main__':
    main()
