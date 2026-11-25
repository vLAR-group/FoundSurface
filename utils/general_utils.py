import torch
import numpy as np
import random
from einops import reduce

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def select_rec_images(images):
    """
    Select a subset of images for reconstruction to fit within GPU memory constraints.
    This function uniformly samples images from the input tensor.

    Args:
        images (torch.Tensor): Input images tensor of shape [B, S, C, H, W].

    Returns:
        torch.Tensor: A tensor containing a subset of images.
    """
    B, S, C, H, W = images.shape
    step = S // 50 + 1
    sub_images_idx = torch.arange(0, S, step)
    return sub_images_idx

def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: tuple (height, width) of the grid
    return:
    pos_embed: [grid_size[0]*grid_size[1], embed_dim] or [n_cls_token+grid_size[0]*grid_size[1], embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)

    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)

    emb = torch.cat([emb_x, emb_y], dim=-1)

    return emb.view(H, W, embed_dim)

def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.float()

def mean_pathify(neighbor_pixel, patch_size):
    """
    Mean-pool a pixel grid into patches.

    Args:
        neighbor_pixel: [B, N, C, H, W]
        patch_size: patch size

    Returns:
        [B, N, C, H // patch_size, W // patch_size]
    """
    neighbor_patch = reduce(
        neighbor_pixel,
        'b n c (h ph) (w pw) -> b n c h w',
        lambda t, axes: t.mean(dim=axes),
        ph=patch_size[0],
        pw=patch_size[1]
    )
    return neighbor_patch

def search_nearest_tokens(neighbor_patchified_pixel, neighbor_size, neighbor_patch_size, topk=5):
    neighbor_patchified_pixel = neighbor_patchified_pixel.flatten(-2, -1).transpose(-1, -2)
    B, N, L, D = neighbor_patchified_pixel.shape

    neighbor_patch_x, neighbor_patch_y = torch.meshgrid(
    torch.arange(0, neighbor_size[1], device=neighbor_patchified_pixel.device),
    torch.arange(0, neighbor_size[0], device=neighbor_patchified_pixel.device),
    indexing='xy'
    )
    neighbor_patch_x = (neighbor_patch_x + 0.5) * neighbor_patch_size[1]
    neighbor_patch_y = (neighbor_patch_y + 0.5) * neighbor_patch_size[0]
    neighbor_view_patch_center = torch.stack([neighbor_patch_x, neighbor_patch_y], dim=-1)
    neighbor_view_patch_center = neighbor_view_patch_center.flatten(0, 1).unsqueeze(0)

    _, K, __ = neighbor_view_patch_center.shape

    neighbor_patchified_pixel_flat = neighbor_patchified_pixel.view(B*N, L, D)

    dist_flat = torch.cdist(neighbor_patchified_pixel_flat, neighbor_view_patch_center, p=2)

    dist = dist_flat.view(B, N, L, K)

    dist_merged = dist.transpose(1, 2).reshape(B, L, N*K)

    _, min_idx = torch.topk(dist_merged, topk, dim=-1, largest=False, sorted=False)

    return min_idx

def search_ray_nearest_tokens(neighbor_patchified_pixel, neighbor_size, neighbor_patch_size, topk=5):
    neighbor_patchified_pixel = neighbor_patchified_pixel.transpose(-1, -2)
    B, N, L, D = neighbor_patchified_pixel.shape

    neighbor_patch_x, neighbor_patch_y = torch.meshgrid(
    torch.arange(0, neighbor_size[1], device=neighbor_patchified_pixel.device),
    torch.arange(0, neighbor_size[0], device=neighbor_patchified_pixel.device),
    indexing='xy'
    )
    neighbor_patch_x = (neighbor_patch_x + 0.5) * neighbor_patch_size[1]
    neighbor_patch_y = (neighbor_patch_y + 0.5) * neighbor_patch_size[0]
    neighbor_view_patch_center = torch.stack([neighbor_patch_x, neighbor_patch_y], dim=-1)
    neighbor_view_patch_center = neighbor_view_patch_center.flatten(0, 1).unsqueeze(0)

    _, K, __ = neighbor_view_patch_center.shape

    neighbor_patchified_pixel_flat = neighbor_patchified_pixel.view(B*N, L, D)

    dist_flat = torch.cdist(neighbor_patchified_pixel_flat, neighbor_view_patch_center, p=2)

    dist = dist_flat.view(B, N, L, K)

    dist_merged = dist.transpose(1, 2).reshape(B, L, N*K)

    _, min_idx = torch.topk(dist_merged, topk, dim=-1, largest=False, sorted=False)

    return min_idx

def torch_unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim,
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index

def get_patch_index(H, W, h1, w1, batch_size, device):
    patch_size_x = W // w1
    patch_size_y = H // h1
    grid_x, grid_y = torch.meshgrid(
    torch.arange(0, patch_size_x, device=device),
    torch.arange(0, patch_size_y, device=device),
    indexing='xy'
    )
    grid_x = grid_x.repeat_interleave(h1, dim=0).repeat_interleave(w1, dim=1)
    grid_y = grid_y.repeat_interleave(h1, dim=0).repeat_interleave(w1, dim=1)
    patch_idx = (grid_y * patch_size_x + grid_x).flatten().unsqueeze(0).repeat(batch_size, 1)
    batch_idx = torch.arange(0, batch_size, device=device).unsqueeze(-1).repeat(1, H*W)
    patch_idx = patch_idx + batch_idx * (patch_size_x * patch_size_y)

    return patch_idx

import logging
def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.

    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside
                                  [-hard_max, hard_max] will be clamped. If None,
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor

    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor
