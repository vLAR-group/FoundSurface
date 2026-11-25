import torch
import torch.nn as nn
EPS = 1e-6


def complete_ray_metric(pred, gt, mask=None):
    if mask is None:
        mask = gt > 0.1
    else:
        mask = mask & (gt > 0.1)

    thresh = torch.maximum((gt[mask] / (pred[mask] + EPS)), (pred[mask] / gt[mask]))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()
    ade = torch.mean(torch.abs(pred[mask] - gt[mask]))
    rmse = torch.sqrt(torch.mean(torch.square(pred[mask] - gt[mask])))
    rel_ade = (torch.abs(pred - gt)[mask] / gt[mask])
    rel_ade = torch.mean(rel_ade)
    sq_rel_ade = (torch.square(pred - gt)[mask] / gt[mask])
    sq_rel_ade = torch.mean(sq_rel_ade)
    return ade, rmse, rel_ade, sq_rel_ade, a1, a2, a3


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute the mean angular error between predicted and reference normals
    """
    dot_products = torch.sum(gt * pred, dim=-1)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    mae = torch.acos(dot_products)
    return mae

class NormalMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth normal maps.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def forward(self, pred, gt):
        angle_err = torch.rad2deg(mean_angular_error(pred, gt))
        mae = torch.mean(angle_err)
        med_err = torch.median(angle_err)
        pct_gt_5 = 100.0 * torch.sum(angle_err < 11.25, axis=0) / angle_err.shape[0]
        pct_gt_10 = 100.0 * torch.sum(angle_err < 22.5, axis=0) / angle_err.shape[0]
        pct_gt_30 = 100.0 * torch.sum(angle_err < 30, axis=0) / angle_err.shape[0]

        return mae.item(), med_err.item(), pct_gt_5.item(), pct_gt_10.item(), pct_gt_30.item()
