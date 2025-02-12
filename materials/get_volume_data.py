import os
import numpy as np
import scipy
from scipy import interpolate, ndimage
from bisect import bisect_left
from cavasspy.ops import read_cavass_file, get_voxel_spacing
from jhammer.io import read_txt_2_list
from jhammer.io import save_json
from tqdm import tqdm

# from materials.meas_stds_const import WDS_MEAN, WDS_STD
# "AxR"
labels = ["AxR"]
test_val = ["test"]
ct_image_path = "/volume/volume_image_json"
data_path = "/BIM/label_GT/AxR/"

WDS_MEAN = 955.297506302056
WDS_STD = 222.9325617386932

two = [1,2,4,8,16,32,64,128,256,512]

def takeCloseset(list,num):
    if num > list[-1] or num < list[0]:
        return list[-1]
    pos = bisect_left(list, num)
    if pos == 0:
        return list[0]
    if pos == len(list):
        return list[-1]
    before = list[pos -1]
    after = list[pos]
    if after - num <num - before:
        return after
    else:
        return before

for label in labels:
    for t_v in test_val:

        samples = read_txt_2_list(f"/3D/{label}_{t_v}_samples.txt")
        saved_path = f"/volume/volume_{label}_json"
        for ct_name in tqdm(samples):
            saved_file_name = f"{ct_name}.json"
            image_path = os.path.join(ct_image_path, saved_file_name)
            im0_file = os.path.join('/IM0/Test_bca', f"{ct_name}.IM0")
            im0_data = read_cavass_file(im0_file)
            image_data = {"image": im0_data,
                          "subject": ct_name,
                          "mean": WDS_MEAN,
                          "std": WDS_STD}

            if not os.path.exists(image_path):
                save_json(image_path, image_data)
