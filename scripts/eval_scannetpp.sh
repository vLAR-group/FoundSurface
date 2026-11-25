#!/usr/bin/env bash
# Scannet++ (Scannetpp) evaluation
cd "$(dirname "$0")/.."

STAGE1="${STAGE1:-/path/to/stage1/dpt_ckpt_0000099.pth}"
STAGE2="${STAGE2:-/path/to/stage2/dpt_ckpt_0000049.pth}"
DATA="${DATA:-/path/to/Scannetpp/Scannetpp_nerf}"

# CUDA_VISIBLE_DEVICES=$1 python test_foundsurface.py --model_path "$STAGE1" \
#                         --test_dataset "Scannetpp(ROOT='$DATA', resolution=[(640, 480)], rec_views=50, split='test')" \
#                         --raylet_path "$STAGE2" \
#                         --surface_patch_size 8

# Surface-normal evaluation (paper setting); uncomment if needed:
CUDA_VISIBLE_DEVICES=$1 python test_foundsurface.py --model_path "$STAGE1" \
                        --test_dataset "Scannetpp(ROOT='$DATA', resolution=[(640, 480)], rec_views=50, split='test')" \
                        --raylet_path "$STAGE2" \
                        --surface_patch_size 8 --normal
