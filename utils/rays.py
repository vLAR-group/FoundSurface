import torch
from einops import rearrange
EPS = 1e-8
import numpy as np

def compute_rays(c2w, K, h=None, w=None, device="cuda"):
    """
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        K (torch.tensor): [b, v, 3, 3]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w]
        ray_d (torch.tensor): [b, v, 3, h, w]
    """

    b, v = c2w.size()[:2]
    c2w = c2w.reshape(b * v, 4, 4)

    fx, fy, cx, cy = K[:, :, 0, 0], K[:, :, 1, 1], K[:, :, 0, 2], K[:, :, 1, 2]
    h_orig = int(2 * cy.max().item())
    w_orig = int(2 * cx.max().item())
    if h is None or w is None:
        h, w = h_orig, w_orig

    if h_orig != h or w_orig != w:
        fx = fx * w / w_orig
        fy = fy * h / h_orig
        cx = cx * w / w_orig
        cy = cy * h / h_orig

    K = K.reshape(b * v, 3, 3)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(device), x.to(device)
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    x = (x + 0.5 - K[:, 0, 2:3]) / K[:, 0, 0:1]
    y = (y + 0.5 - K[:, 1, 2:3]) / K[:, 1, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

    return ray_o, ray_d


def get_posed_input(images=None, ray_o=None, ray_d=None, method="default_plucker"):
    '''
    Args:
        images: [b, v, c, h, w]
        ray_o: [b, v, 3, h, w]
        ray_d: [b, v, 3, h, w]
        method: Method for creating pose conditioning
    Returns:
        posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
    '''
    if method == "custom_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        pose_cond = torch.cat([ray_d, nearest_pts], dim=2)

    elif method == "aug_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)

    else:
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

    if images is None:
        return pose_cond
    else:
        return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)


def geometric_structure(neighbor_c2ws, neighbor_Ks, novel_K, novel_size, neighbor_size, novel_patch_size, neighbor_patch_size):
    novel_patch_x, novel_patch_y = torch.meshgrid(
        torch.arange(0, novel_size[1], device=neighbor_c2ws.device),
        torch.arange(0, novel_size[0], device=neighbor_c2ws.device),
        indexing='xy'
    )
    novel_patch_x = (novel_patch_x + 0.5) * novel_patch_size[1]
    novel_patch_y = (novel_patch_y + 0.5) * novel_patch_size[0]
    novel_view_patch_center = torch.stack([novel_patch_x, novel_patch_y], dim=-1)
    homo_novel_view_patch_center = torch.cat([novel_view_patch_center, torch.ones_like(novel_view_patch_center[..., :1])], dim=-1)
    neighbor_patch_x, neighbor_patch_y = torch.meshgrid(
        torch.arange(0, neighbor_size[1], device=neighbor_c2ws.device),
        torch.arange(0, neighbor_size[0], device=neighbor_c2ws.device),
        indexing='xy'
    )
    neighbor_patch_x = (neighbor_patch_x + 0.5) * neighbor_patch_size[1]
    neighbor_patch_y = (neighbor_patch_y + 0.5) * neighbor_patch_size[0]
    neighbor_view_patch_center = torch.stack([neighbor_patch_x, neighbor_patch_y], dim=-1)
    homo_neighbor_view_patch_center = torch.cat([neighbor_view_patch_center, torch.ones_like(neighbor_view_patch_center[..., :1])], dim=-1)
    R = neighbor_c2ws[..., :3, :3]
    t = neighbor_c2ws[..., :3, 3:4]

    t_x, t_y, t_z = t[..., 0, 0], t[..., 1, 0], t[..., 2, 0]
    zero = torch.zeros_like(t_x)
    t_skew = torch.stack([
        torch.stack([zero, -t_z, t_y], dim=-1),
        torch.stack([t_z, zero, -t_x], dim=-1),
        torch.stack([-t_y, t_x, zero], dim=-1)
    ], dim=-2)
    E = t_skew @ R

    neighbor_Ks_inv = torch.linalg.inv(neighbor_Ks).transpose(-1, -2)
    novel_K_inv = torch.linalg.inv(novel_K)
    F = neighbor_Ks_inv @ E @ novel_K_inv
    epiolar_line = F @ homo_novel_view_patch_center.reshape(-1, 3).transpose(0, 1).unsqueeze(0).unsqueeze(0)
    epiolar_line = epiolar_line.transpose(-1, -2)

    numerator = torch.abs(epiolar_line @ homo_neighbor_view_patch_center.reshape(-1, 3).transpose(0, 1).unsqueeze(0).unsqueeze(0))
    denominator = torch.norm(epiolar_line[..., :2] + 1e-6, dim=-1, keepdim=True)
    distance = numerator / (denominator + 1e-6)
    return distance


def cam_sph2coord(sph):
    if sph.shape[-1] == 2:
        r = 1.0
    else:
        r = sph[..., 2]
    theta = sph[..., 0]
    phi = sph[..., 1]
    coord = torch.stack([torch.sin(theta) * torch.cos(phi) * r,
                         torch.cos(theta) * r,
                         torch.sin(theta) * torch.sin(phi) * r], dim=-1)
    return coord


def cam_coord2sph(coord, normalize = False, return_radius=False):
    xz_dist = torch.sqrt(coord[..., 2] ** 2 + coord[..., 0] ** 2)
    theta = torch.arctan2(xz_dist, coord[..., 1] + EPS)
    phi = torch.arctan2(coord[..., 2], (coord[..., 0]) + EPS)

    if normalize:
        theta = 2. * (theta / torch.pi) - 1.
        phi = 2. * (phi / torch.pi) - 1.
    if not return_radius:
        return torch.stack([theta, phi], -1)
    else:
        rad = torch.sqrt((coord**2).sum(-1))
        return torch.stack([theta, phi, rad], -1)

def gradient(y, x, grad_outputs=None, create_graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=create_graph, allow_unused=True)[0]
    return grad


def get_surface_normal(t, raydirs, slice_range = None):
    dt = gradient(t, raydirs)
    if slice_range is not None:
        raydirs = raydirs[slice_range[0] : slice_range[1]]
        dt = dt[slice_range[0] : slice_range[1]]
    raydirs = raydirs.squeeze()
    dt = dt.squeeze()
    dtdtheta, dtdphi = dt[..., :1], dt[..., 1:]
    sin_theta, cos_theta = torch.sin(raydirs[..., :1]), torch.cos(raydirs[..., :1])
    sin_phi, cos_phi = torch.sin(raydirs[..., 1:]), torch.cos(raydirs[..., 1:])
    dtheta = torch.cat([(dtdtheta * sin_theta + t * cos_theta) * cos_phi,
                        (dtdtheta * cos_theta - t * sin_theta),
                        (dtdtheta * sin_theta + t * cos_theta) * sin_phi,], dim=-1)

    dphi = torch.cat([(dtdphi * cos_phi - t * sin_phi) * sin_theta,
                    (dtdphi * cos_theta),
                    (dtdphi * sin_phi + t * cos_phi) * sin_theta], dim=-1)

    normal = torch.cross(dphi, dtheta, dim=-1)
    normal = normal / (torch.linalg.norm(normal+EPS, dim=-1, keepdim=True)+EPS)
    return normal

def get_rays(K, use_pixel_centers = True):
    H, W = K[1, 2]*2, K[0, 2]*2
    pixel_center = .5 if use_pixel_centers else 0.
    x, y = np.meshgrid(np.arange(W, dtype=np.float32) + pixel_center,
                           np.arange(H, dtype=np.float32) + pixel_center)
    ray_d_cam = np.stack([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], K[2, 2] * np.ones_like(x)], -1)
    rays_d = ray_d_cam / (np.linalg.norm(ray_d_cam, axis=-1, keepdims=True) + EPS)

    return rays_d

def get_rays_torch(K, use_pixel_centers = True):
    H, W = K[1, 2]*2, K[0, 2]*2
    pixel_center = .5 if use_pixel_centers else 0.
    x, y = torch.meshgrid(torch.arange(W, dtype=torch.float32) + pixel_center,
                           torch.arange(H, dtype=torch.float32) + pixel_center, indexing='xy')
    ray_d_cam = torch.stack([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], K[2, 2] * torch.ones_like(x)], -1)
    rays_d = ray_d_cam / (torch.linalg.norm(ray_d_cam, dim=-1, keepdim=True) + EPS)

    return rays_d


def dpt2normal(depth, K=None, c2w_pose=None):

    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    K = [[1, 0], [0, 1]] if K is None else K
    fx, fy = abs(K[0][0]), abs(K[1][1])

    dz_dv, dz_du = np.gradient(depth)
    du_dx = fx / (depth + EPS)
    dv_dy = fy / (depth + EPS)

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    normal_unit = normal_cross / np.linalg.norm(normal_cross + EPS, axis=2, keepdims=True)
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    if c2w_pose is not None:
        normal_unit = np.sum(normal_unit[..., np.newaxis, :] * c2w_pose[..., :3, :3], -1)

    return normal_unit

def depth_to_ray_distance(
    depth_maps: np.ndarray,
    fx: float,
    fy: float = None,
    cx: float = None,
    cy: float = None,
    eps: float = 1e-8
) -> np.ndarray:

    H, W = depth_maps.shape

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v, indexing='xy')

    x_norm = (u_grid - cx) / (fx + eps)
    y_norm = (v_grid - cy) / (fy + eps)

    scale_factor = np.sqrt(x_norm**2 + y_norm**2 + 1.0)

    ray_dists = depth_maps * scale_factor

    return ray_dists


from torch import Tensor
from typing_extensions import List, Optional, Tuple
import cv2

def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> Tensor:
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()

    return image_coords

def get_means3d_backproj(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, List]:
    """Backprojection using camera intrinsics and extrinsics

    image_coords -> (x,y,depth) -> (X, Y, depth)

    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)
    if depths.device != device:
        depths = depths.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)

    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy
    means3d[:, 2] = depths[:, 0]

    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]

    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords

def pcd_to_normal(xyz: Tensor):
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
    top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = xyz[..., 1 : hd - 1, 2:wd, :]
    left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)
    return xyz_normal

def normal_from_depth_image(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    smooth: bool = False,
):
    """estimate normals from depth map"""
    if smooth:
        if torch.count_nonzero(depths) > 0:
            print("Input depth map contains 0 elements, skipping smoothing filter")
        else:
            kernel_size = (9, 9)
            depths = torch.from_numpy(
                cv2.GaussianBlur(depths.cpu().numpy(), kernel_size, 0)
            ).to(device)
    means3d, _ = get_means3d_backproj(depths, fx, fy, cx, cy, img_size, c2w, device)
    means3d = means3d.view(img_size[1], img_size[0], 3)
    normals = pcd_to_normal(means3d)
    return normals


import torch

def depth_to_ray_distance_torch(
    depth_maps: torch.Tensor,
    fx: float,
    fy: float = None,
    cx: float = None,
    cy: float = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Convert a depth map (distance along the optical axis) to ray distance.

    Args:
        depth_maps: Tensor of shape (..., H, W), depth values along the optical axis.
        fx: Focal length in x direction (in pixels).
        fy: Focal length in y direction (if None, same as fx).
        cx: Principal point x coordinate (in pixels). If None, must be provided.
        cy: Principal point y coordinate (in pixels). If None, must be provided.
        eps: Small constant to avoid division by zero.

    Returns:
        ray_dists: Tensor of same shape as depth_maps, distances along each ray.
    """
    if fy is None:
        fy = fx
    if cx is None or cy is None:
        raise ValueError("cx and cy must be provided.")

    H, W = depth_maps.shape[-2], depth_maps.shape[-1]
    device = depth_maps.device
    dtype = depth_maps.dtype

    fx = torch.tensor(fx, device=device, dtype=dtype)
    fy = torch.tensor(fy, device=device, dtype=dtype)
    cx = torch.tensor(cx, device=device, dtype=dtype)
    cy = torch.tensor(cy, device=device, dtype=dtype)

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

    x_norm = (u_grid - cx) / (fx + eps)
    y_norm = (v_grid - cy) / (fy + eps)

    scale_factor = torch.sqrt(x_norm**2 + y_norm**2 + 1.0)

    ray_dists = depth_maps * scale_factor

    return ray_dists
