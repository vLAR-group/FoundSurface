import math
import numpy as np
import torch

def resize_intrinsics(intrinsics, original_size, target_size):
    """
    Resize camera intrinsics to match a new image size.

    Args:
        intrinsics (torch.Tensor): (3, 3) or (B, 3, 3) camera intrinsics.
        original_size (tuple): original (W, H).
        target_size (tuple): target (W, H).

    Returns:
        torch.Tensor: resized intrinsics with the same shape as the input.
    """
    sw = target_size[0] / original_size[0]
    sh = target_size[1] / original_size[1]

    resized_k = intrinsics.clone()

    resized_k[..., 0, 0] *= sw
    resized_k[..., 1, 1] *= sh

    resized_k[..., 0, 2] *= sw
    resized_k[..., 1, 2] *= sh

    return resized_k

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def convert_d(d, focal, out='dist'):
    H, W = d.shape[0], d.shape[1]
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    L = np.sqrt(np.power(j - H / 2., 2) + np.power(i - W / 2., 2) + focal ** 2)
    fl = focal / L
    if out == 'dist':
        return d / fl
    elif out == 'dep':
        return d * fl
    else:
        raise NotImplementedError


def convert_d_torch(d, focal, out='dist'):
    H, W = d.shape[-2], d.shape[-1]
    device, dtype = d.device, d.dtype
    i, j = torch.meshgrid(torch.arange(W, device=device, dtype=dtype),
                        torch.arange(H, device=device, dtype=dtype), indexing='xy')
    L = torch.sqrt((j - H / 2.)** 2 + (i - W / 2.)** 2 + focal ** 2)
    fl = focal / L
    if out == 'dist':
        return d / fl
    elif out == 'dep':
        return d * fl
    else:
        raise NotImplementedError

def create_pt3d(depth, K, c2w, new_world_transform):
    """
    Convert depth maps to 3D points in the world coordinate system.

    Args:
        depth: Depth maps of shape [B, S, H, W]
        K: Intrinsic matrices of shape [B, S, 3, 3]
        c2w: Camera-to-world matrices of shape [B, S, 4, 4]
        new_world_transform: New world pose matrix of shape [B, 4, 4]
    Returns:
        pts3d: 3D points in the world coordinate system of shape [B, S, H, W, 3]
    """
    B, S, H, W = depth.shape

    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    x = x[None, None, :, :].expand(B, S, -1, -1)
    y = y[None, None, :, :].expand(B, S, -1, -1)
    cam_coord = torch.stack([(x - K[..., 0:1, 2:3]) * depth / K[..., 0:1, 0:1], (y - K[..., 1:2, 2:3]) * depth / K[..., 1:2, 1:2], depth, torch.ones_like(x)], -1)
    cam_coord = cam_coord.reshape(B, S, H * W, 4)
    world_coord = cam_coord @ c2w.permute(0, 1, 3, 2)
    pts3d = world_coord @ new_world_transform.unsqueeze(1).permute(0, 1, 3, 2)
    pts3d = pts3d[..., :3]
    pts3d = pts3d.reshape(B, S, H, W, 3)

    return pts3d


def preprocess_images(rec_images, rec_depths, org_rec_Ks, target_resolution = (392, 518)):
    ''' Preprocess images, depths, and intrinsic matrices for model input.
        Resize the input to target resolution.
    Args:
        rec_images (torch.Tensor): Input images of shape [B, S, C, H, W].
        rec_depths (torch.Tensor): Depth maps of shape [B, S, H, W].
        rec_Ks (torch.Tensor): Intrinsic matrices of shape [B, S, 3, 3].
        target_resolution (tuple): Target resolution for resizing, default is (392, 518).
    Returns:
        rec_images (torch.Tensor): Preprocessed images of shape [B, S, C, H', W'].
        rec_depths (torch.Tensor): Preprocessed depth maps of shape [B, S, H', W'].
        rec_Ks (torch.Tensor): Preprocessed intrinsic matrices of shape [B, S, 3, 3].
    '''
    b, s, c, h, w = rec_images.shape
    rec_images = rec_images.reshape(b * s, c, h, w)
    rec_depths = rec_depths.reshape(b * s, h, w)
    rec_images = torch.nn.functional.interpolate(
        rec_images, size=target_resolution, mode='bicubic', align_corners=False
    )
    rec_depths = torch.nn.functional.interpolate(
        rec_depths.unsqueeze(1), size=target_resolution, mode='nearest'
    ).squeeze(1)

    rec_images = rec_images.reshape(b, s, c, target_resolution[0], target_resolution[1])
    rec_depths = rec_depths.reshape(b, s, target_resolution[0], target_resolution[1])

    rec_Ks = org_rec_Ks.clone()
    rec_Ks[:, :, 0, 0] *= target_resolution[1] / w
    rec_Ks[:, :, 1, 1] *= target_resolution[0] / h
    rec_Ks[:, :, 0, 2] *= target_resolution[1] / w
    rec_Ks[:, :, 1, 2] *= target_resolution[0] / h
    return rec_images, rec_depths, rec_Ks


def depth2xyzmap(depth, K):
    """
    Convert depth map to XYZ map in camera coordinates.

    Args:
        depth: Depth map of shape [B, S, H, W]
        K: Intrinsic matrix of shape [B, S, 3, 3]

    Returns:
        xyz_map: XYZ map of shape [B, S, H, W, 3]
    """
    B, S, H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    x = x[None, None, :, :].expand(B, S, -1, -1)
    y = y[None, None, :, :].expand(B, S, -1, -1)
    xyz_map = torch.stack([(x - K[..., 0:1, 2:3]) * depth / K[..., 0:1, 0:1],
                           (y - K[..., 1:2, 2:3]) * depth / K[..., 1:2, 1:2],
                           depth], -1)
    return xyz_map
