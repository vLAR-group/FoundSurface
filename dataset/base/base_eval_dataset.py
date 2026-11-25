# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
import PIL
import numpy as np
import torch

from dataset.base.easy_dataset import EasyDataset
from dataset.utils.transforms import ImgNorm
import dataset.utils.cropping as cropping


class BaseEvalDataset (EasyDataset):
    """ Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(self, *,
                 split="test",
                 resolution=None,
                 transform=ImgNorm,
                 aug_crop=False,
                 seed=None,
                 novel_view_num = 10,
                 rec_views = 50,
                 fix_res = False,):
        self.split = split
        self.vggt_target_size = 518
        self.novel_view_num = novel_view_num
        self.rec_views = rec_views
        self._set_resolutions(resolution)
        self.fix_res = fix_res

        self.transform = transform
        if isinstance(transform, str):
            transform = eval(transform)

        self.aug_crop = aug_crop
        self.seed = seed

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = '[' + ';'.join(f'{w}x{h}' for w, h in self._resolutions) + ']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()
            self._rng = np.random.default_rng(seed=seed)

        resolution = self._resolutions[ar_idx]
        rec_data_dict, novel_data_dict, scene_info = self._get_views(idx, resolution, self._rng)

        for key in rec_data_dict:
            rec_data_dict[key] = torch.stack(rec_data_dict[key], dim=0)
        for key in novel_data_dict:
            novel_data_dict[key] = torch.stack(novel_data_dict[key], dim=0)


        return rec_data_dict, novel_data_dict, scene_info

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        W, H = image.size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            if rng.integers(2):
                resolution = resolution[::-1]

        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """ returns (is_good, err_msg)
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)

        view['K'] = view['camera_intrinsics'][[1, 0, 2]]
