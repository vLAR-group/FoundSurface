import open3d as o3d
import numpy as np
import warnings


def save_pts(pts, path, m=None, color=None, normal=None):
    if m is not None:
        pts = pts[m[...,0]==1]
    pts = np.reshape(pts, [-1, 3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array(color).reshape(-1, 3))
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.array(normal).reshape(-1, 3))
    if pts.shape[0] > 5000000:
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        warnings.warn("Warning: Point cloud has more than 5 million points, downsampling to reduce size.")
    o3d.io.write_point_cloud(path, pcd)

def convert_dpt2pt(depth, K, pose):
    H, W = depth.shape[0], depth.shape[1]
    x, y = np.meshgrid(np.arange(W, dtype=np.float32),
                            np.arange(H, dtype=np.float32))
    cam_pt = np.stack([(x - K[0, 2]) * depth / K[0, 0], (y - K[1, 2]) * depth / K[1, 1], K[2, 2] * np.ones_like(x) * depth, np.ones_like(x)], -1)
    pt = (pose[None, None, :, :]@cam_pt[:, :, :, None]).squeeze()[:, :, :3]
    return pt

import torch
def convert_dpt2pt_torch(depth: torch.Tensor, K: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """
    Convert a depth map to a point cloud in world coordinates.

    Args:
        depth: (H, W) depth values, float32/float64.
        K: (3, 3) camera intrinsics.
        pose: (4, 4) camera-to-world transform.

    Returns:
        pt: (H, W, 3) point cloud in world coordinates.
    """
    H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    x = torch.arange(W, dtype=dtype, device=device)
    y = torch.arange(H, dtype=dtype, device=device)
    xx = x.unsqueeze(0).expand(H, W)
    yy = y.unsqueeze(1).expand(H, W)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    kz = K[2, 2]

    cam_x = (xx - cx) * depth / fx
    cam_y = (yy - cy) * depth / fy
    cam_z = kz * depth
    ones = torch.ones_like(cam_x)

    cam_pts = torch.stack([cam_x, cam_y, cam_z, ones], dim=-1)

    cam_pts_4 = cam_pts.unsqueeze(-1)
    world_pts_homo = pose @ cam_pts_4
    world_pts = world_pts_homo.squeeze(-1)[..., :3]

    return world_pts

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention(weights, layer=0):
    """
    weights: numpy array, shape (num_heads, L, L)
    query_tokens: list of str, length L
    key_tokens: list of str, length L
    """
    for head in range(weights.shape[0]):
        attn = weights[head]
        if attn.max()<0.3:
            vmax= 0.1
        else:
            vmax = 0.3
        plt.figure(figsize=(6, 6))
        sns.heatmap(attn, cmap='viridis', xticklabels=100, yticklabels=100, square=False, cbar=True, vmin=0, vmax=vmax)
        plt.title(f'Layer {layer} Head {head} Attention')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.savefig(f'./debug_visualization/twostage/short_seq/attention_layer{layer}_head{head}.png')

def point2depth(points, w2c, K, img_h, img_w):
    """
    points: (N, 3)
    w2c: (4, 4)
    K: (3, 3)
    img_h, img_w: int
    """
    pcd = o3d.t.geometry.PointCloud(points.reshape(-1, 3).cpu().numpy())
    pcd = pcd.to(o3d.core.Device("CUDA:0"))
    world_to_cam = o3d.core.Tensor(w2c.squeeze().cpu().numpy(), dtype=o3d.core.Dtype.Float32).to(o3d.core.Device("CUDA:0"))
    intrinsic_tensor = o3d.core.Tensor(K.cpu().numpy(), dtype=o3d.core.Dtype.Float32).to(o3d.core.Device("CUDA:0"))
    depth_map = pcd.project_to_depth_image(
        width=img_w,
        height=img_h,
        intrinsics=intrinsic_tensor,
        extrinsics=world_to_cam,
        depth_scale=1.0,
        depth_max=20
    )
    depth_map = np.asarray(depth_map.cpu())
    depth_map = torch.from_numpy(depth_map).float()
    return depth_map


def to_normalmap(x, m=None, white_bkgd=True):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    m = m if m is not None else np.ones_like(x)

    o = 255. * np.ones_like(x) if white_bkgd else np.zeros_like(x)
    xm = x[m[...,0]==1]
    o[m[...,0]==1] = 255 * (xm + 1.) / 2.
    return o.astype(np.uint8)
