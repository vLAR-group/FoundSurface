
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = x.clone()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce
    else:
        return x
