from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from jhammer.samplers.coordinate_generator import GridCoordinateGenerator
from jhammer.samplers.utils import central_crop, get_margin


class GridSampler:
    def __init__(self, data: Union[np.ndarray, torch.Tensor], patch_size, valid_size=None):
        """
        Perform grid sample and recovery. The last N dimensions of given data will be split.

        Args:
            data (numpy.ndarray or torch.Tensor): The data needs to be sample.
            patch_size (sequence): The data will be divided into patches with the shape of patch_shape.
            valid_size (sequence or None, optional, default=None): Valid patch size, used for restore. If `None`, as
                same as `patch_size`.
        """

        self.patch_size = np.asarray(patch_size, dtype=np.uint32)
        self.valid_size = np.asarray(valid_size, dtype=np.uint32) if valid_size is not None else self.patch_size
        assert len(self.patch_size.shape) == len(self.valid_size.shape)

        self.full_size = data.shape
        # shape of data, but batch and/or channel dimensions are excluded
        self.original_shape = self.full_size[-len(self.patch_size):]
        self.coordinate_generator = GridCoordinateGenerator(self.original_shape, patch_shape=self.patch_size,
                                                            valid_shape=self.valid_size)

        if not (self.patch_size == self.valid_size).all():
            if isinstance(data, torch.Tensor):
                pad_size = tuple(np.flip(self.coordinate_generator.padding_size, axis=0).flatten())
                self.padded_data = F.pad(data, pad=pad_size, mode="constant", value=0)
            else:
                pad_size = self.coordinate_generator.padding_size
                if len(self.full_size) != len(self.valid_size):
                    extra_dimensions = [[0, 0]] * (len(self.full_size) - len(self.valid_size))
                    pad_size = np.concatenate((np.asarray(extra_dimensions, dtype=np.int32), pad_size))
                self.padded_data = np.pad(data, pad_width=pad_size, mode="constant", constant_values=0)
        else:
            self.padded_data = data
        self.index = 0

    def restore(self, patches, restore_shape=None):
        """
        Every element's batch/channel dimensions in blocks should be put at the beginning.

        Args:
            patches (sequence):
            restore_shape (sequence or None, optional, default=None): Restore array shape. Batch/channel dimensions
                of restored array can differ from the data which was divided as long as keeping the same image
                dimensions. If None, as same as `self.full_size`.
        """

        patches = list(map(lambda x: self._shrink_shape(x), patches))
        if not restore_shape:
            restore_shape = self.full_size
        return self._rebuild_image(patches, restore_shape)

    def _shrink_shape(self, data: Union[np.ndarray, torch.Tensor]):
        """
        Shrink data to assigned shape.

        Args:
            data (numpy.ndarray or torch.Tensor):

        Returns:

        """

        if (self.patch_size == self.valid_size).all():
            return data
        center = [e // 2 if e % 2 != 0 else e // 2 - 1 for e in self.valid_size]
        center = np.asarray(center)
        center = center + self.coordinate_generator.padding_size[:, 0]
        cropped = central_crop(data, center, self.valid_size)
        return cropped

    def _rebuild_image(self, patches, restore_shape):
        assert len(patches) > 0
        assert len(self.coordinate_generator.valid_central_coordinates) == len(patches)
        result = np.zeros(shape=restore_shape, dtype=patches[0].dtype) if isinstance(patches[0], np.ndarray) \
            else torch.zeros(size=restore_shape, dtype=patches[0].dtype, device=patches[0].device)

        block_margin = get_margin(self.valid_size)
        margin = block_margin[:, 0][None, :]
        coordinates = self.coordinate_generator.valid_central_coordinates - margin
        extra_dimensions = [slice(None)] * (len(restore_shape) - len(self.valid_size))
        for i, block in enumerate(patches):
            slice_sequence = extra_dimensions + \
                             [slice(coordinates[i, j], coordinates[i, j] + self.valid_size[j])
                              for j in range(len(self.valid_size))]

            result[tuple(slice_sequence)] = block
        return result

    def __len__(self):
        return len(self.coordinate_generator)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration

        patch = central_crop(self.padded_data, self.coordinate_generator[self.index], self.patch_size)
        self.index += 1
        return patch
