import numpy as np
import cv2
from einops import rearrange

from transforms.transforms import Transform


# class ToType(Transform):
#     def __init__(self, keys, dtype):
#         super().__init__(keys)
#         self.dtype = dtype
#
#     def _call_fun(self, data, *args, **kwargs):
#         for key in self.keys:
#             value = data[key].astype(self.dtype)
#             data[key] = value
#         return data

class ToType(Transform):
    def __init__(self, keys, dtype):
        super().__init__(keys)
        self.dtype = dtype

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            if isinstance(data[key], np.ndarray):
                value = data[key].astype(self.dtype)
            else:
                value = np.array(data[key]).astype(self.dtype)
            data[key] = value
        return data

class Rearrange(Transform):
    def __init__(self, keys, pattern):
        """
        Change the arrangement of given elements.

        Args:
            keys (str or sequence):
            pattern (str): Arranging pattern. For example "i j k -> j k i".
        """

        super().__init__(keys)
        self.pattern = pattern

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            value = data[key]
            value = rearrange(value, self.pattern)
            data[key] = value
        return data

class AddChannel(Transform):
    def __init__(self, keys, dim):
        """
        Add additional dimension in specific position.

        Args:
            keys (str or sequence):
            dim (int):
        """

        super().__init__(keys)
        self.dim = dim

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            value = data[key]
            value = np.expand_dims(value, axis=self.dim)
            data[key] = value
        return data

class ZscoreNormalization(Transform):
    def __init__(self, keys, mean_value=None, std_value=None, mean_key=None, std_key=None):
        """
        Perform z-score normalization. `mean_key` and `std_key` indicate keys of mean and std value in data. You can
        also set common mean and std values for all data. Mean and std values provided by each sample will be used
        firstly if they exist.

        Args:
            keys (str or sequence):
            mean_value (float or None, optional, default=None):
            std_value (float or None, optional, default=None):
            mean_key (str or None, optional, default=None):
            std_key (str or None, optional, default=None):
        """

        super().__init__(keys)
        self.mean_value = mean_value
        self.std_value = std_value
        self.mean_key = mean_key
        self.std_key = std_key

    def _call_fun(self, data, *args, **kwargs):
        mean = data[self.mean_key] if self.mean_key in data else self.mean_value
        std = data[self.std_key] if self.std_key in data else self.std_value
        assert mean and std

        for key in self.keys:
            value = data[key]
            value = (value - mean) / std # value: (512, 512, 234)
            data[key] = value
        return data


class MinMaxNormalization(Transform):
    def __init__(self, keys, lower_bound_percentile=1, upper_bound_percentile=99):
        """
        Perform min-max normalization.

        Args:
            keys (str or sequence):
            lower_bound_percentile (int, optional, default=1):
            upper_bound_percentile (int, optional, default=99):
        """

        super().__init__(keys)
        self.lower_bound_percentile = lower_bound_percentile
        self.upper_bound_percentile = upper_bound_percentile

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            image = data[key]
            min_value, max_value = np.percentile(image, (self.lower_bound_percentile, self.upper_bound_percentile))
            image = (image - min_value) / (max_value - min_value)
            data[key] = image
        return data


class GetShape(Transform):
    def __init__(self, keys):
        """
        Get array shape.

        Args:
            keys (str or sequence):
        """

        super().__init__(keys)

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            shape = data[key].shape
            shape = np.asarray(shape)
            data[f"{key}_shape"] = shape
        return data

# ###################################################################### Resize #########################################################

class ResizeImgAndLab(Transform):
    def __init__(self, keys, target_size ,orientation):
        super().__init__(keys)
        self.target_size = target_size
        self.orientation = orientation

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            value = data[key]

            if self.orientation == "Transverse":
                # target_size = [128, 128, 70]
                resized_img = np.zeros((self.target_size[0], self.target_size[1], self.target_size[2]),dtype=np.float32)
                resize_lab = np.zeros((self.target_size[0], self.target_size[1], self.target_size[2]), dtype=np.uint8)
                if key == "AxR" or key == "LCR":
                    for d in range(value.shape[2]):
                        resize_lab[:, :, d] = cv2.resize(value[:, :, d].astype(np.uint8),(self.target_size[1], self.target_size[0]))
                        data[key] = resize_lab.astype(bool)
                        print(f"resize_lab_type:", data[key].dtype)
                        print("resize_lab:", data[key].shape)
                else:
                    for d in range(value.shape[2]):
                        resized_img[:, :, d] = cv2.resize(value[:, :, d].astype(float), (self.target_size[0], self.target_size[1]))
                        data[key] = resized_img
                        print(f"resized_img_type:", data[key].dtype)
                        print("resized_img:", data[key].shape)

            # 如果值是一个单独的切片数组
            if self.orientation == "Coronal":
                resized_img = np.zeros((value.shape[0], self.target_size[0], self.target_size[1]),dtype=np.float32)
                resize_lab = np.zeros((value.shape[0], self.target_size[0], self.target_size[1]), dtype=np.uint8)
                # print("resized_img_zeros_Coronal:", resized_img.shape)
                if key == "AxR" or key == "LCR":
                    for h in range(value.shape[0]):
                        resize_lab[h, :, :] = cv2.resize(value[h, :, :].astype(np.uint8),(self.target_size[1], self.target_size[0]))
                        data[key] = resize_lab.astype(bool)
                else:
                    for h in range(value.shape[0]):
                        resized_img[h, :, :] = cv2.resize(value[h, :, :].astype(float), (self.target_size[1], self.target_size[0]))
                        data[key] = resized_img

            if self.orientation == "Sagittal":
                resized_img = np.zeros((self.target_size[0], value.shape[1], self.target_size[1]),dtype=np.float32)
                resize_lab = np.zeros((self.target_size[0], value.shape[1], self.target_size[1]),dtype=np.float32)
                # print("resized_img_zeros_Sagittal:", resized_img.shape)
                if key == "AxR" or key == "LCR":
                    for w in range(value.shape[1]):
                        resize_lab[:, w, :] = cv2.resize(value[:, w, :].astype(np.uint8),(self.target_size[1], self.target_size[0]))
                        data[key] = resize_lab.astype(bool)
                else:
                    for w in range(value.shape[1]):
                        # print("value.shape[1]", value.shape[1])
                        resized_img[:, w, :] = cv2.resize(value[:, w, :].astype(float),(self.target_size[1], self.target_size[0]))
                        # print("resized_img:", resized_img.shape)
                        data[key] = resized_img
                # print("value:",value.shape)
                # print("value[0]:",value[0].shape)

        return data