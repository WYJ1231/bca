import numpy as np
import torch
from medpy.metric.binary import __surface_distances


def dsc(input_data: torch.Tensor, target: torch.Tensor):
    eps = 1e-5
    input_class = input_data.flatten()
    target_class = target.flatten()
    intersection = (input_class * target_class).sum()
    union = input_class.sum() + target_class.sum()
    dice = (2. * intersection + 1e-5) / (union + eps)
    return dice


def hd(result, reference, voxel_spacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxel_spacing, connectivity)
    hd2 = __surface_distances(reference, result, voxel_spacing, connectivity)
    hd = max(hd1.max(), hd2.max())
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd, hd95
