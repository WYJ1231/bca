from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from .utils import get_margin, encode_one_hot


class CoordinateGenerator(ABC):

    @abstractmethod
    def __getitem__(self, index):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def regenerate(self):
        ...

    @staticmethod
    # # python 3.10
    # def get_margin_mask(image_shape: tuple | list, patch_size: tuple | list):
    def get_margin_mask(image_shape: Union[tuple, list], patch_size: Union[tuple, list]):
        """
        Build a mask that cover the area which can be view as the central point of the patch with the size of
        patch_size. Mask the central area that can be extracted. Left the margin area.

        Args:
            image_shape (sequence):
            patch_size (sequence):

        Returns:

        """

        assert len(image_shape) == len(patch_size)
        margin = get_margin(patch_size)
        mask = np.zeros(image_shape, dtype=bool)
        mask[tuple([slice(j[0], i - j[1]) for i, j in zip(image_shape, margin)])] = 1
        return mask


class WeightedCoordinateGenerator(CoordinateGenerator):

    def __init__(self, num_coordinates, patch_size, weight_map):
        """
        Choose coordinates according to the weight.

        Args:
            num_coordinates (int):
            patch_size (sequence):
            weight_map (numpy.ndarray or torch.Tensor):
        """

        # number of coordinates that need to be picked.
        self.n = num_coordinates
        self.patch_size = patch_size
        self.weight_map = weight_map
        self.coordinates = self.get_coordinates()

    def __getitem__(self, index):
        return self.coordinates[index]

    def __len__(self):
        return self.n

    def get_coordinates(self):
        weight_map = self.weight_map / np.sum(self.weight_map).astype(np.float64)
        weight_map = weight_map.flatten()
        coordinates = np.random.choice(weight_map.size, size=self.n, p=weight_map)
        coordinates = np.asarray(np.unravel_index(coordinates, self.weight_map.shape))
        return coordinates.T

    def regenerate(self):
        self.coordinates = self.get_coordinates()


class BalancedCoordinateGenerator(WeightedCoordinateGenerator):
    def __init__(self, num_coordinates, data, patch_size):
        """
        Generate coordinates according to the proportion of categories. In general, the smaller the percentage of
        category, the more likely it is to be taken. Parameter data is the un-one-hot label.

        Args:
            num_coordinates (int):
            data (numpy.ndarray or torch.Tensor): The label matrix. this data will be needed to generate the weight map.
            patch_size (sequence):
        """

        self.n = num_coordinates
        self.patch_size = patch_size
        super().__init__(num_coordinates=num_coordinates, patch_size=patch_size, weight_map=self.get_weight_map(data))

    def get_weight_map(self, data):
        # one-hot first
        weight_map = encode_one_hot(data)

        # exclude margin
        margin_mask = self.get_margin_mask(data.shape, self.patch_size)
        weight_map = weight_map * margin_mask
        weight_map = weight_map.astype(np.float32)

        # normalize on every category axis
        category_sum = np.sum(weight_map, axis=tuple(range(1, len(data.shape) + 1)), keepdims=True)
        weight_map = weight_map / (category_sum + 1)
        weight_map = np.sum(weight_map, axis=0).astype(np.float32)
        return weight_map


class GridCoordinateGenerator(CoordinateGenerator):
    def __init__(self, original_shape, patch_shape, valid_shape):
        """

        Args:
            original_shape (sequence): The shape of original image.
            patch_shape (sequence): The shape of one patch.
            valid_shape (sequence): The size of valid patch. Valid shape should be less than patch shape. A valid patch
                needs to be extracted from a patch.
        """

        self.original_shape = np.asarray(original_shape, dtype=np.uint32)
        self.patch_shape = np.asarray(patch_shape, dtype=np.uint32)
        self.valid_shape = np.asarray(valid_shape, dtype=np.uint32)

        self.padding_size = get_margin(patch_shape) - get_margin(valid_shape)
        self.valid_central_coordinates = self.get_coordinates()
        self.padded_coordinates = self.valid_central_coordinates + self.padding_size[:, 0][None, :]

    def get_coordinates(self):
        """
        Get central coordinates according to the original shape and valid shape.

        Returns:

        """

        coordinate_dims = [range(0, i, j) for i, j in zip(self.original_shape, self.valid_shape)]
        # coordinate_dims = [range(0, i, j) for i, j in zip(self.original_shape, [1, 512, 512])]
        coordinates = np.meshgrid(*coordinate_dims, indexing="ij")
        coordinates = list(map(lambda x: x.flatten(), coordinates))
        # get the corner coordinate of each patch
        coordinates = np.stack(coordinates, axis=-1)
        coordinates = np.where(coordinates + self.valid_shape > self.original_shape,
                               self.original_shape - self.valid_shape, coordinates)

        # shift coordinate to patch center
        shift = get_margin(self.valid_shape)[:, 0]
        central_coordinates = coordinates + shift
        # print("self.central_coordinates:", central_coordinates.astype(np.int32))
        return central_coordinates.astype(np.int32)

    def __getitem__(self, index):
        return self.padded_coordinates[index]

    def __len__(self):
        return len(self.padded_coordinates)

    def regenerate(self):
        ...
