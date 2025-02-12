import json
import os.path
from base64 import b64encode, b64decode
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
from numpy.lib.format import dtype_to_descr, descr_to_dtype


def read_mat(input_file, key="scene"):
    from scipy.io import loadmat
    data = loadmat(input_file)[key]
    return data


def save_mat(output_file, data, key="scene"):
    from scipy.io import savemat
    ensure_dir(output_file)
    savemat(output_file, {key: data})


def read_txt_2_list(input_file):
    with open(input_file, "r") as input_file:
        return [each.strip("\n") for each in input_file.readlines()]


def write_list_2_txt(output_file, data_lst):
    ensure_dir(output_file)
    with open(output_file, "w") as file:
        for i in range(len(data_lst)):
            file.write(data_lst[i])
            if i != len(data_lst) - 1:
                file.write("\n")


def save_nii(output_file,
             data,
             voxel_spacing=None,
             orientation="LPI"):
    """
    Save image with nii format.

    Args:
        output_file (str or pathlib.Path):
        data (numpy.ndarray):
        voxel_spacing (sequence or None, optional, default=None): `tuple(x, y, z)`. Voxel spacing of each axis. If None,
            make `voxel_spacing` as `(1.0, 1.0, 1.0)`.
        orientation (str, optional, default="LPI"): "LPI" | "ARI". LPI: Left-Posterior-Inferior;
            ARI: Anterior-Right-Inferior.

    Returns:

    """

    import nibabel as nib
    ensure_dir(output_file)
    if voxel_spacing is None:
        voxel_spacing = (1.0, 1.0, 1.0)  # replace this with your desired voxel spacing in millimeters

    # if orientation == "LPI":
    #     affine_matrix = np.diag(list(voxel_spacing) + [1.0])
    # elif orientation == "ARI":
    #     # calculate the affine matrix based on the desired voxel spacing and ARI orientation
    #     affine_matrix = np.array([
    #         [0, -voxel_spacing[0], 0, 0],
    #         [-voxel_spacing[1], 0, 0, 0],
    #         [0, 0, voxel_spacing[2], 0],
    #         [0, 0, 0, 1]
    #     ])
    # else:
    #     raise ValueError(f"Unsupported orientation {orientation}.")
    # match orientation:
    #     case "LPI":
    #         affine_matrix = np.diag(list(voxel_spacing) + [1.0])
    #     case "ARI":
    #         # calculate the affine matrix based on the desired voxel spacing and ARI orientation
    #         affine_matrix = np.array([
    #             [0, -voxel_spacing[0], 0, 0],
    #             [-voxel_spacing[1], 0, 0, 0],
    #             [0, 0, voxel_spacing[2], 0],
    #             [0, 0, 0, 1]
    #         ])
    #     case _:
    #         raise ValueError(f"Unsupported orientation {orientation}.")

    if orientation == "LPI":
        affine_matrix = np.diag(list(voxel_spacing) + [1.0])
    elif orientation == "ARI":
        affine_matrix = np.array([
            [0, -voxel_spacing[0], 0, 0],
            [-voxel_spacing[1], 0, 0, 0],
            [0, 0, voxel_spacing[2], 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError(f"Unsupported orientation {orientation}.")

    # create a NIfTI image object
    nii_img = nib.Nifti1Image(data, affine=affine_matrix)
    nib.save(nii_img, output_file)


def read_dicom_series(input_dir: Union[str, Path]):
    from pydicom import dcmread
    if not os.path.exists(input_dir):
        raise FileExistsError(f"{input_dir} does not exist.")
    instances = []
    for each in os.listdir(input_dir):
        if each.endswith(".dcm"):
            instances.append(each)

    instances.sort()
    images = []
    for slice_file_name in instances:
        slice_file = os.path.join(input_dir, slice_file_name)
        dicom_data = dcmread(slice_file)
        if "PixelData" in dicom_data:
            pixel_data = dicom_data.pixel_array
            images.append(pixel_data)

    return np.stack(images)


# JSON
def np_object_hook(dct):
    """
    Convert JSON list or scalar to numpy.

    Args:
        dct (mapping):

    Returns:

    """

    if "__ndarray__" in dct:
        shape = dct["shape"]
        dtype = descr_to_dtype(dct["dtype"])
        if shape:
            order = "C" if dct["Corder"] else "F"
            if dct["base64"]:
                np_obj = np.frombuffer(b64decode(dct["__ndarray__"]), dtype=dtype)
                np_obj = np_obj.copy(order=order)
            else:
                np_obj = np.asarray(dct["__ndarray__"], dtype=dtype, order=order)
            return np_obj.reshape(shape)

        if dct["base64"]:
            np_obj = np.frombuffer(b64decode(dct["__ndarray__"]), dtype=dtype)[0]
        else:
            t = getattr(np, dtype.name)
            np_obj = t(dct["__ndarray__"])
        return np_obj

    return dct


def read_json(input_json_file):
    """
    Read and convert `input_json_file`.

    Args:
        input_json_file (str or pathlib.Path):

    Returns:

    """

    if not os.path.exists(input_json_file):
        raise FileExistsError(f"{input_json_file} does not exist.")
    with open(input_json_file, 'r') as json_file:
        dct = json.load(json_file, object_hook=np_object_hook)
    return dct


class NumpyJSONEncoder(json.JSONEncoder):
    def __init__(self, primitive=False, base64=True, **kwargs):
        """
        JSON encoder for `numpy.ndarray`.

        Args:
            primitive (bool, optional, default=False): Use primitive type if `True`. In primitive schema,
                `numpy.ndarray` is stored as JSON list and `np.generic` is stored as a number.
            base64 (bool, optional, default=True): Use base64 to encode.
            **kwargs:
        """

        self.primitive = primitive
        self.base64 = base64
        super().__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            if self.primitive:
                return obj.tolist()
            else:
                if self.base64:
                    data_json = b64encode(obj.data if obj.flags.c_contiguous else obj.tobytes()).decode("ascii")
                else:
                    data_json = obj.tolist()
                dct = OrderedDict(__ndarray__=data_json,
                                  dtype=dtype_to_descr(obj.dtype),
                                  shape=obj.shape,
                                  Corder=obj.flags["C_CONTIGUOUS"],
                                  base64=self.base64)
                return dct
        return super().default(obj)


def save_json(output_file, obj, primitive=False, base64=True):
    """
    Convert obj to JSON object and save as file.

    Args:
        output_file (str or pathlib.Path):
        obj (mapping):
        primitive (bool, optional, default=False): Use primitive type if `True`. In primitive schema, `numpy.ndarray` is
            stored as JSON list and `np.generic` is stored as a number.
        base64 (bool, optional, default=True): Use base64 to encode.

    Returns:

    """

    ensure_dir(output_file)
    with open(output_file, "w") as file:
        json.dump(obj, file, cls=NumpyJSONEncoder, **{"primitive": primitive, "base64": base64})


def ensure_dir(input_file):
    input_dir = os.path.split(input_file)[0]
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
