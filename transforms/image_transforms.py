from abc import ABC
import numpy as np
from typing import Tuple, Union

from transforms import Transform


class ImageAugmentTransform(Transform, ABC):
    def __init__(self, keys, prob):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data, *args, **kwargs):
        if np.random.random() < self.prob:
            data = self._call_fun(data)
        return data


class RandFlip(ImageAugmentTransform):
    def __init__(self, keys, prob, spatial_axis=None):
        """
        Flip image along specific spatial axis.

        Args:
            keys (str or sequence):
            prob (float):
            spatial_axis (int or sequence or None): Flip the array along the assigned axis. If more than one axes are
                given, flip the array along each axis. If None, flip the data along all axes.
        """

        super().__init__(keys, prob)
        self.axis = spatial_axis

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            image = data[key]
            flipped_elem_data = np.flip(image, axis=self.axis)
            data[key] = flipped_elem_data.copy()
        return data


class RandRotate90(ImageAugmentTransform):
    # def __init__(self, keys, prob, max_k=1, spatial_axes: tuple[int, int] = (0, 1)):
    def __init__(self, keys, prob, max_k=1, spatial_axes: Tuple[int, int] = (0, 1)):
        """
        Rotate array by 90 degrees in the plan specified by spatial axes. For a 2-D array with `k = 1` and
        `axes = (0, 1)`, the rotation direction will be counterclockwise.

        Args:
            keys (str or sequence):
            prob (float):
            max_k (int, optional, default=1):
            spatial_axes (sequence, optional, default=(0, 1)): Two int numbers, rotate the array in the plane
                determined by these axes.
        """

        super().__init__(keys, prob)
        self.max_k = max_k
        self.axes = spatial_axes

    def _call_fun(self, data, *args, **kwargs):
        k = np.random.randint(1, self.max_k + 1)
        for key in self.keys:
            image = data[key]
            rotated_elem_data = np.rot90(image, k=k, axes=self.axes)
            data[key] = rotated_elem_data
        return data


class RandScaleIntensity(ImageAugmentTransform):
    # # >3.10
    # def __init__(self, keys, prob, factors: tuple[float, float] | float):
    def __init__(self, keys, prob, factors: (Tuple[float, float], float)):
        """
        Scale the image intensity using factor by `i = i * (1 + factor)`.

        Args:
            keys (str or sequence):
            prob (float):
            factors (float or sequence): If provided factors is a number, the factor will be uniformly sampled from
                [-factors, factors). If factors is a pair of numbers, the factor will be picked from
                [min(factors), max(factors)].
        """

        super().__init__(keys, prob)
        if isinstance(factors, (int, float)):
            self.factors = (min(-factors, factors), max(-factors, factors))
        elif len(factors) != 2:
            raise ValueError(f"factors should be a number or a pair of numbers, got {factors}")
        else:
            self.factors = (min(factors), max(factors))

    def _call_fun(self, data, *args, **kwargs):
        factor = np.random.uniform(self.factors[0], self.factors[1])
        for key in self.keys:
            image = data[key]
            scaled_data = image * (1. + factor)
            data[key] = scaled_data
        return data


class RandShiftIntensity(ImageAugmentTransform):
    # def __init__(self, keys, prob, offsets: tuple[float, float] | float):
    def __init__(self, keys, prob, offsets: Union[Tuple[float, float], float]):
        """
        Scale the image intensity using factor by `i = i * (1 + factor)`.

        Args:
            keys (str or sequence):
            prob (float):
            offsets (float or sequence): If provided offset is a number, the offset will be uniformly picked from
                [-offsets, offsets). If offsets is a pair of numbers, the offsets will be picked from
                [min(offsets), max(offsets)].
        """

        super().__init__(keys, prob)
        if isinstance(offsets, (int, float)):
            self.offsets = (min(-offsets, offsets), max(-offsets, offsets))
        elif len(offsets) != 2:
            raise ValueError(f"offsets should be a number or a pair of numbers, got {offsets}")
        else:
            self.factors = (min(offsets), max(offsets))

    def _call_fun(self, data, *args, **kwargs):
        offset = np.random.uniform(self.offsets[0], self.offsets[1])
        for key in self.keys:
            image = data[key]
            scaled_data = image + offset
            data[key] = scaled_data
        return data
