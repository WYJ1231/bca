import random
from abc import ABC, abstractmethod
from itertools import cycle

from jhammer.samplers.coordinate_generator import BalancedCoordinateGenerator
from jhammer.samplers.patch_picker import WeightedPatchPicker


class PreloadedDataset(ABC):
    def __init__(self, samples: list,
                 patch_size,
                 n_patches_per_sample,
                 n_samples_alive=1,
                 shuffle=True,
                 transforms=None):
        """
        Store the subjects on a queue of length queue_length in the RAM.

        Args:
            samples (list): List of samples.
            patch_size (sequence):
            n_patches_per_sample (int): Number of patches cropped from each sample.
            n_samples_alive (int, optional, default=1): Maximum number of samples saved in RAM, should in the range of
                [1, len(samples)].
            shuffle (bool, optional, default=True): If `True`, shuffle the samples.
            transforms (torchvision.transforms.Compose or None):
        """

        if shuffle:
            random.shuffle(samples)
        self.sample_iterator = cycle(samples)

        self.patch_size = patch_size
        self.n_patches_per_sample = n_patches_per_sample

        if n_samples_alive <= 0:
            n_samples_alive = 1
        elif n_samples_alive > len(samples):
            n_samples_alive = len(samples)
        self.n_samples_alive = n_samples_alive

        self.sample_queue = [None, ] * self.n_samples_alive
        self.index = 0

        self.transforms = transforms

    def update_queue(self, pos: int):
        """
        Update the queue element in the given position.

        Args:
            pos (int):

        Returns:

        """

        sample_idx = next(self.sample_iterator)

        if self.sample_queue[pos] and self.sample_queue[pos][0] == sample_idx:
            # only update the patch picker.
            _, patch_picker = self.sample_queue[pos]
            patch_picker.reset()
        else:
            data = self.get_sample(sample_idx)
            coordinate_generator = BalancedCoordinateGenerator(self.n_patches_per_sample,
                                                               self.get_weight_map(data),
                                                               self.patch_size)

            patch_picker = WeightedPatchPicker(data, self.patch_size, coordinate_generator)
            self.sample_queue[pos] = (sample_idx, patch_picker)

    @abstractmethod
    def get_sample(self, index):
        """
        Get sample by index from `self.sample_iterator`.

        Args:
            index (object): Object like str or int that indices the sample.

        Returns:

        """

        ...

    @abstractmethod
    def get_weight_map(self, data):
        ...

    def __iter__(self):
        return self

    def __next__(self):
        index = self.index
        self.index = (self.index + 1) % self.n_samples_alive

        # only invoke when visiting the index at the first time.
        if self.sample_queue[index] is None:
            self.update_queue(index)
        try:
            data = self.next(index)
        except StopIteration:
            self.update_queue(index)
            data = self.next(index)
        return data

    def next(self, index):
        _, patch_picker = self.sample_queue[index]
        data = next(patch_picker)
        if self.transforms:
            data = self.transforms(data)
        return data
