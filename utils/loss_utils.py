import torch
import math
import torch.distributed as dist
import torch.nn as nn
from utils.misc import get_world_size


EPS = 1e-8
def l1_loss_mask(network_output, gt, mask):
    return (torch.abs((network_output - gt)*mask)).sum() / (mask.sum() + EPS)

def rel_l1_loss_mask(network_output, gt, mask):
    return (torch.abs((network_output - gt)*mask) / (gt + EPS)).sum() / (mask.sum() + EPS)

def l1_conf_mask(network_output, gt, mask, conf, alpha=0.2):
    return ((torch.abs(network_output - gt)*conf - alpha*torch.log(conf))*mask).sum() / (mask.sum() + EPS)

def l2_loss_mask(network_output, gt, mask):
    return (torch.square((network_output - gt)*mask)).sum() / (mask.sum() + EPS)

def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.tensor(0.0, device=prediction.device)
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def adjust_learning_rate(optimizer, cur_epoch, warmup_epochs, epochs, set_lr, min_lr):
    """Decay the learning rate with half-cycle cosine after warmup"""

    if cur_epoch < warmup_epochs:
        lr = set_lr * cur_epoch / warmup_epochs
    else:
        lr = min_lr + (set_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (cur_epoch - warmup_epochs) / (epochs - warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = x.clone()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce
    else:
        return x

class StableCauchyLoss(nn.Module):
    def __init__(self, s=1.0, reduction='mean'):
        super(StableCauchyLoss, self).__init__()
        self.s = s
        self.reduction = reduction

    def forward(self, network_output, gt, mask):
        residual = (torch.abs((network_output - gt)*mask)).sum() / (mask.sum() + EPS)
        scaled_residual_sq = (residual / self.s).pow(2)
        loss = torch.log1p(scaled_residual_sq)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
