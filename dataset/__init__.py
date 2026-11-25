from .utils.transforms import *
from utils.misc import get_world_size, get_rank
from .base.batched_sampler import BatchedRandomSampler
from .scannet import ScanNet, ScannetEval
from .BlendedMVS import BlendedMVS
from .ARKitScenes import ARKitScenes
from .Hypersim import Hypersim
from .GTA_SfM import GTA_SfM
from .MVS_Synth import MVS_Synth
from .WildRGBD import WildRGBD
from .S3DIS import S3DIS
from .SceneNN import SceneNN
from .Scannetpp import Scannetpp
from .Multiscan import Multiscan
from .Replica import Replica
from .SevenScenes import SevenScenes
from .Tanks_Temples import Tnt
from .MuSHRoom import MuSHRoom


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch

    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,
    )

    return data_loader


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader
