#!/usr/bin/env bash
# Inference-speed benchmark: per-view latency, FPS and GPU memory.
#
#   bash scripts/inference_speed.sh [gpu_id]
#
# Needs the torch_knnquery CUDA extension (used for voxelization here):
#   pip install ./thirdparty/torch_knnquery
set -euo pipefail

GPU_ID="${1:-0}"
shift || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE1="${STAGE1:-$ROOT/ckpt/stage1/dpt_ckpt_0000099.pth}"
STAGE2="${STAGE2:-$ROOT/ckpt/stage2/dpt_ckpt_0000049.pth}"
DATA="${DATA:-/path/to/Scannet/scannet_nerf/scans_test}"

cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python inference.py \
    --model_path "$STAGE1" \
    --raylet_path "$STAGE2" \
    --test_dataset "ScannetEval(ROOT='$DATA', resolution=[(640, 480)], rec_views=50)" \
    --surface_patch_size 8 \
    --num_workers 8 \
    "$@"
