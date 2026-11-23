#!/usr/bin/env bash
# Add KITScenes Multimodal inference to an official UniAD (v1) checkout.
#
#   git clone https://github.com/OpenDriveLab/UniAD.git && git -C UniAD checkout main
#   bash tools/kitscenes/setup_uniad.sh /path/to/UniAD
#
# Copies the KITScenes dataset adapter, config and tools from this repository into
# UniAD and applies a small patch (dataset registration and test-time guards for
# inputs without GT). Tested on UniAD main (532fc33) and a3bd2f7. UniAD v2.0
# (the current default branch) uses a different code base and is not supported.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 /path/to/UniAD" >&2
  exit 1
fi
UNIAD=$(cd "$1" && pwd)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DMAD=$(cd "$HERE/../.." && pwd)

if [ ! -f "$UNIAD/projects/configs/stage2_e2e/base_e2e.py" ]; then
  echo "error: $UNIAD does not look like a UniAD v1 checkout (no projects/configs/stage2_e2e/base_e2e.py)." >&2
  echo "       Check out the main branch: git -C $UNIAD checkout main" >&2
  exit 1
fi
if [ -e "$UNIAD/projects/mmdet3d_plugin/datasets/kitscenes_multimodal_dataset.py" ]; then
  echo "error: KITScenes support is already installed in $UNIAD." >&2
  exit 1
fi

git -C "$UNIAD" apply --check "$HERE/uniad/uniad_kitscenes.patch"
git -C "$UNIAD" apply "$HERE/uniad/uniad_kitscenes.patch"

cp "$DMAD/projects/mmdet3d_plugin/datasets/kitscenes_multimodal_dataset.py" \
   "$UNIAD/projects/mmdet3d_plugin/datasets/"
cp "$HERE/uniad/uniad_kitscenes_multimodal.py" "$UNIAD/projects/configs/stage2_e2e/"
mkdir -p "$UNIAD/tools/kitscenes"
cp "$HERE"/{inference,prepare_dataset_json,export_jsonl,visualize}.py "$UNIAD/tools/kitscenes/"

echo "KITScenes Multimodal inference installed into $UNIAD"
echo "config: projects/configs/stage2_e2e/uniad_kitscenes_multimodal.py (see docs/KITSCENES.md)"
