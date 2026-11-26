<p align="center">
  <h1 align="center" style="font-size: 1.5rem;">FoundSurface - <span style="color: #dc2626; font-size: 0.7em;">A Foundation Surface Reconstruction Model</span> </h1>
  <p align="center">
    <strong>NeurIPS 2026</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2611.34128"><img src="https://img.shields.io/badge/arXiv-2611.34128-b31b1b.svg" alt="arXiv"></a>
    <a href="https://vlar-group.github.io/FoundSurface.html"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="https://vlar-group.github.io/FoundSurface/static/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-green" alt="Paper"></a>
    <a href="https://github.com/vLAR-group/FoundSurface"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="Code"></a>
    <a href="https://drive.google.com/drive/folders/1T-L7KuJuVyTzn0EpnpcR8iomA1baRMeh?usp=sharing"><img src="https://img.shields.io/badge/Checkpoints-Google%20Drive-yellow" alt="Checkpoints"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="License"></a>
  </p>
</p>

> **Note:** this repository contains the **test code** of FoundSurface. Training code is not part of this release.

FoundSurface reconstructs 3D scene surfaces from unposed images in a single
feed-forward pass. For every pixel of a target view it predicts the distance
travelled by that ray, giving a compact ray-distance field from which surfaces
and analytical normals follow directly.

## Installation

Python 3.10 and CUDA 11.8 (one GPU is enough for the default 50-view setting).

```bash
# torch first, then the CUDA-specific wheels, then the rest
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install spconv-cu118==2.3.8
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
pip install -r requirements.txt

# only for inference.py (speed benchmark)
pip install --no-build-isolation ./thirdparty/torch_knnquery
```

`spconv` and `pytorch3d` must match the installed torch/CUDA. `pytorch3d` and
`torch_knnquery` are compiled during installation, so the CUDA 11.8 toolkit
(`nvcc`) must be available.

## Checkpoints

Download the trained checkpoints from
[Google Drive](https://drive.google.com/drive/folders/1T-L7KuJuVyTzn0EpnpcR8iomA1baRMeh?usp=sharing):

```
FoundSurface_checkpoints/
├── stage1_rec_net/dpt_ckpt_0000099.pth     # stage-1 rec-net / sampler (791 MiB)
└── stage2_raylet/dpt_ckpt_0000049.pth      # stage-2 raylet (630 MiB)
```

The VGGT-1B backbone is third-party and not included. Place it at
`vggt/vggt_model.pt`:

```bash
huggingface-cli download facebook/VGGT-1B model.pt --local-dir vggt
mv vggt/model.pt vggt/vggt_model.pt
```

Only `aggregator.*`, `camera_head.*` and `point_head.*` are loaded.

Run everything **from the repository root**: the script loads
`./vggt/vggt_model.pt` through a hard-coded relative path.

## Data

```
<ROOT>/<scene>/transforms_train.json     # reconstruction views (always this file)
<ROOT>/<scene>/transforms_test.json      # target views (or transforms_all.json)
```

`split=` in the dataset constructor only selects which file provides the target
views; reconstruction frames always come from `transforms_train.json`. Each JSON
contains `depth_scale` and `frames`, and every frame provides `file_path`,
`depth_file_path`, `fx`, `fy`, `cx`, `cy` and a NeRF-style c2w
`transform_matrix`.

Raw datasets can be downloaded from their project websites.

**Test datasets**

- [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/)
- [ScanNet](https://github.com/ScanNet/ScanNet)
- [Multiscan](https://github.com/smartscenes/multiscan)
- [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Replica (NICE-SLAM / DN-Splatter)](https://cvg-data.inf.ethz.ch/nice-slam/data/)
- [MuSHRoom](https://github.com/TUTvision/MuSHRoom)
- [Tanks and Temples](https://www.tanksandtemples.org/)

Replica scenes are processed following
[DN-Splatter](https://github.com/maturk/dn-splatter).

**Training datasets**

- [ScanNet](https://github.com/ScanNet/ScanNet)
- [Hypersim](https://github.com/apple/ml-hypersim)
- [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
- [GTA-SfM](https://github.com/HKUST-Aerial-Robotics/Flow-Motion-Depth)
- [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
- [WildRGBD](https://github.com/wildrgbd/wildrgbd)
- [S3DIS / 2D-3D-S](http://buildingparser.stanford.edu/dataset.html)
- [SceneNN](https://hkust-vgd.github.io/scenenn)
- [ARKitScenes](https://github.com/apple/ARKitScenes)

Scripts that convert these datasets to the NeRF layout above are provided in
`data_script/`. For BlendedMVS, Hypersim and S3DIS, run the corresponding
`*_process.py` first and then the `*_convert2seq.py` pass. Note that
`S3DIS_process.py` expects the rendered **Stanford 2D-3D-S** release
(`Area_X/data/{rgb,depth,pose}`), not the raw S3DIS point-cloud zips.

## Evaluation

Set the checkpoint and dataset paths, then run the ScanNet++ script with a GPU
id:

```bash
STAGE1=/path/to/FoundSurface_checkpoints/stage1_rec_net/dpt_ckpt_0000099.pth \
STAGE2=/path/to/FoundSurface_checkpoints/stage2_raylet/dpt_ckpt_0000049.pth \
DATA=/path/to/Scannetpp/Scannetpp_nerf \
bash scripts/eval_scannetpp.sh 0
```

or run the entry directly:

```bash
CUDA_VISIBLE_DEVICES=0 python test_foundsurface.py \
  --model_path  /path/to/stage1_rec_net/dpt_ckpt_0000099.pth \
  --raylet_path /path/to/stage2_raylet/dpt_ckpt_0000049.pth \
  --test_dataset "Scannetpp(ROOT='/path/to/Scannetpp/Scannetpp_nerf', resolution=[(640, 480)], rec_views=50, split='test')" \
  --surface_patch_size 8
```

To evaluate several datasets at once, concatenate them with `+`:

```bash
--test_dataset "Scannetpp(ROOT='/path/to/Scannetpp/Scannetpp_nerf', resolution=[(640, 480)], rec_views=50, split='test') \
              + Multiscan(ROOT='/path/to/Multiscan/Multiscan_nerf/test', resolution=[(640, 480)], rec_views=50) \
              + SevenScenes(ROOT='/path/to/7Scenes/7scenes_nerf', resolution=[(640, 480)], rec_views=50) \
              + Replica(ROOT='/path/to/Replica/Replica_nerf', resolution=[(608, 352)], rec_views=50) \
              + MuSHRoom(ROOT='/path/to/MuSHRoom/MuSHRoom_nerf', resolution=[(640, 368)], rec_views=50) \
              + Tnt(ROOT='/path/to/TNT/tnt_nerf', resolution=[(544, 288)], rec_views=50) \
              + ScannetEval(ROOT='/path/to/Scannet/scannet_nerf/scans_test', resolution=[(640, 480)], rec_views=50)"
```

Notes:

- `resolution` only sets the coarse stage-1 grid; the GT ray map and the raylet
  refinement are computed at the **original image resolution** (e.g. 960x720 for
  ScanNet++).
- Dataset strings are `eval()`-ed Python. Use **single quotes inside** the
  double-quoted shell string (`split='test'`); double quotes are stripped by the
  shell.
- `--surface_patch_size 8` is required for the standard resolutions above; the
  default `10` only divides 640x480.
- For single-scene debugging, set `ONLY_SCENE=<scene>` and use `--num_workers 0`.
- `--save_dist` additionally dumps per-frame prediction/GT arrays.

### Surface normals

Normals are derived analytically from the refined ray-distance field. Uncomment
the `--normal` command at the bottom of `scripts/eval_scannetpp.sh`, or run:

```bash
CUDA_VISIBLE_DEVICES=0 python test_foundsurface.py \
  --model_path  /path/to/stage1_rec_net/dpt_ckpt_0000099.pth \
  --raylet_path /path/to/stage2_raylet/dpt_ckpt_0000049.pth \
  --test_dataset "Scannetpp(ROOT='/path/to/Scannetpp/Scannetpp_nerf', resolution=[(640, 480)], rec_views=50, split='test')" \
  --surface_patch_size 8 --normal
```

`--normal` appends `NORMAL_MAE`, `NORMAL_MED_ERR`, `NORMAL_PTC5/10/30` and their
stds to the result JSON; add `--save_normal` to also write normal maps.

### Outputs

Results are written next to `--model_path`:

```
<dirname(--model_path)>/<dataset>_Raylet_fit_top1_view<rec_views>/<split>/<scene>/
├── test_results_net.json      # per-scene metrics (+ stds)
├── test_per_view_net.json     # per-view metrics
├── pred_dist/*.npy            # with --save_dist (float16)
├── gt_dist/*. npy             # with --save_dist (note the space in the suffix)
└── pred_preview/*.png         # with --save_dist (per-frame normalized)
```

Aggregate a run:

```bash
python tools/summarize_results.py <results-dir> --split test
```

## Inference speed

```bash
python inference.py \
  --model_path  /path/to/stage1_rec_net/dpt_ckpt_0000099.pth \
  --raylet_path /path/to/stage2_raylet/dpt_ckpt_0000049.pth \
  --test_dataset "ScannetEval(ROOT='/path/to/scans_test', resolution=[(640, 480)], rec_views=50)" \
  --surface_patch_size 8 --num_workers 8
```

It prints the preparation time, per-view latency (mean/std), FPS and GPU memory.
This entry uses `torch_knnquery.VoxelGrid`, so install
`thirdparty/torch_knnquery` first.

## Citation

```bibtex
@inproceedings{wei2026foundsurface,
  title     = {FoundSurface: Feedforward Ray-Based 3D Scene Surface Reconstruction from Unposed Images},
  author    = {Shenxing Wei and Jinxi Li and Zihui Zhang and Ajay Kumar and Wei Wang and Bo Yang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

<a id="license"></a>

This code builds on [VGGT](https://github.com/facebookresearch/vggt) and on the
dataset base classes from [DUSt3R](https://github.com/naver/dust3r) / CroCo
(`dataset/base/`, CC BY-NC-SA 4.0). See `LICENSE.txt`; the release and the
checkpoints are for non-commercial research use only.
