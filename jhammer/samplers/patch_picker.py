from collections import OrderedDict
from typing import Union, Iterable

import numpy as np
import torch

from .coordinate_generator import CoordinateGenerator
from .utils import central_crop


class WeightedPatchPicker:
    def __init__(self, data: Union[tuple, list, dict, np.ndarray, torch.Tensor],
                 patch_size,
                 coordinates: CoordinateGenerator):
        """
        An iterable for picking a patch from data. The original data can be a single numpy.ndarray or a set of data.

        Args:
            data (sequence or mapping or numpy.ndarray or torch.Tensor):
            patch_size (sequence):
            coordinates (CoordinateGenerator):
        """

        self.data = data
        self.patch_size = patch_size
        self.index = 0
        self.coordinates = coordinates

    def reset(self):
        self.coordinates.regenerate()
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        index = self.index
        self.index += 1

        if index >= len(self):
            raise StopIteration

        if isinstance(self.data, Iterable):
            # if the data is a bundle of samples, all ndarray data will be cropped according the candidate coordinate.
            if isinstance(self.data, dict):
                results = OrderedDict()
                for k, v in self.data.items():
                    if isinstance(v, np.ndarray):
                        results[k] = central_crop(v, self.coordinates[index], self.patch_size)
                    else:
                        results[k] = v
                return results

            results = []
            for data in self.data:
                if isinstance(data, np.ndarray):
                    results.append(central_crop(data, self.coordinates[index], self.patch_size))
                else:
                    results.append(data)
            return tuple(results)
        return central_crop(self.data, self.coordinates[index], self.patch_size)

    def __len__(self):
        return len(self.coordinates)
