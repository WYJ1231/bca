import random
import os

import numpy as np
from cavasspy.ops import read_cavass_file,get_voxel_spacing
from pandas import read_csv

from jhammer.io import read_json, save_json
from jhammer.io import read_txt_2_list
from tqdm import tqdm

# labels = ["AxR", "LCR"]
labels = ["LCR"]
# orientations = ["Coronal", "Sagittal"]
# orientations = ["Coronal", "Sagittal", "Transverse"]
orientations = ["Transverse"]
data_path = "/W-DS1"

WDS_MEAN = 955.297506302056
WDS_STD = 222.9325617386932

def get_boundary(boundary_file, orientation):
    if boundary_file is None:
        return None
    df = read_csv(boundary_file)
    border_data = np.array(df)
    subject_list = border_data[:, 0]
    inferior_list = border_data[:, 1]
    superior_list = border_data[:, 2]

    boundary_dict = dict()
    if orientation == 'Sagittal':
        inferior_list_2 = border_data[:, 3]
        superior_list_2 = border_data[:, 4]
        for subject, inferior, superior , inferior_2, superior_2 in zip(subject_list, inferior_list, superior_list, inferior_list_2, superior_list_2):
            boundary_dict[subject] = (inferior, superior, inferior_2, superior_2)
    else:
        for subject, inferior, superior in zip(subject_list, inferior_list, superior_list):
            boundary_dict[subject] = (inferior, superior)

    return boundary_dict

for orientation in orientations:

    ct_image_path = f"/{orientation}/images"

    for label in labels:
        slice_label_save_path = f"/{orientation}/{label}"
        boundary_file = f"/{orientation}/{label}_boundary.csv"
        samples = read_txt_2_list(f"/{label}/subject_{label}_training.txt")
        boundary_dict = get_boundary(boundary_file, orientation)
        missing_im0s = []
        all_saved_file_names = []
        saved_file_names = []

        for ct_name in tqdm(samples):

            im0_file = os.path.join(data_path, ct_name, f"{ct_name}.IM0")
            im0_data = None

            label_file = os.path.join(data_path, ct_name, f"{ct_name}_{label}.BIM")
            if not os.path.exists(label_file):
                print(f"label: {label}. label file is missing: {label_file}")
            label_data = read_cavass_file(label_file).astype(bool)
            label_space = get_voxel_spacing(label_file)

            if orientation == 'Transverse' or orientation == 'Coronal':
                # 横切面,冠状面
                if orientation == 'Transverse':
                    inferior, superior = boundary_dict[ct_name]
                    start = inferior - 5 if inferior - 5 >= 0 else 0
                    ends = superior + 6 if superior + 6 < label_data.shape[2] else label_data.shape[2]

                    end = label_data.shape[2]
                    for d in range(1, end):
                        slice_data = label_data[..., d]
                        data = {"label": slice_data, "subject": ct_name, "slice_number": d, "class": label}
                        saved_file_name = f"{ct_name}_{d:0>3d}.json"
                        save_path = os.path.join(slice_label_save_path, saved_file_name)

                        if not os.path.exists(save_path):
                            save_json(save_path, data)

                        all_saved_file_names.append(saved_file_name)

                        image_file = os.path.join(ct_image_path, saved_file_name)
                        if not os.path.exists(image_file):
                            if im0_data is None:
                                im0_data = read_cavass_file(im0_file)
                            image_data = {"image": im0_data[..., d],
                                          "subject": ct_name,
                                          "slice_number": d,
                                          "mean": WDS_MEAN,
                                          "std": WDS_STD}
                            save_json(image_file, image_data)
                    for dd in range(start, ends):
                        saved_file_name = f"{ct_name}_{dd:0>3d}.json"
                        saved_file_names.append(saved_file_name)
                random.shuffle(all_saved_file_names)
                random.shuffle(saved_file_names)
                ALL_txt_file_path = f"/{label}/ALL_{label}_training_samples_{orientation}.txt"
                txt_file_path = f"/txt/{label}/{label}_training_samples_{orientation}.txt"
                with open(ALL_txt_file_path, "w") as txt_file:
                    for file_name in all_saved_file_names:
                        txt_file.write(file_name + "\n")

                with open(txt_file_path, "w") as txt_file:
                    for file_name in saved_file_names:
                        txt_file.write(file_name + "\n")

                if orientation == 'Coronal':
                    inferior, superior = boundary_dict[ct_name]
                    start = inferior - 5 if inferior - 5 >= 0 else 0
                    ends = superior + 6 if superior + 6 < label_data.shape[0] else label_data.shape[0]

                    end = label_data.shape[0]
                    for h in range(1, end):
                        slice_data = label_data[h, ...]
                        data = {"label": slice_data, "subject": ct_name, "slice_number": h, "class": label, "space":label_space}
                        saved_file_name = f"{ct_name}_{h:0>3d}.json"
                        save_path = os.path.join(slice_label_save_path, saved_file_name)

                        if not os.path.exists(save_path):
                            save_json(save_path, data)

                        all_saved_file_names.append(saved_file_name)

                        image_file = os.path.join(ct_image_path, saved_file_name)
                        if not os.path.exists(image_file):
                            if im0_data is None:
                                im0_data = read_cavass_file(im0_file)
                            im0_space = get_voxel_spacing(im0_file)
                            image_data = {"image":im0_data[h, ...],
                                          "subject": ct_name,
                                          "slice_number": h,
                                          "space":im0_space,
                                          "mean": WDS_MEAN,
                                          "std": WDS_STD}
                            save_json(image_file, image_data)
                    for hh in range(start, ends):
                        saved_file_name = f"{ct_name}_{hh:0>3d}.json"
                        saved_file_names.append(saved_file_name)
                random.shuffle(all_saved_file_names)
                random.shuffle(saved_file_names)
                All_txt_file_path = f"/{label}/ALL_{label}_training_samples_{orientation}.txt"
                txt_file_path = f"/{label}/{label}_training_samples_{orientation}.txt"
                with open(All_txt_file_path, "w") as txt_file:
                    for file_name in all_saved_file_names:
                        txt_file.write(file_name + "\n")

                with open(txt_file_path, "w") as txt_file:
                    for file_name in saved_file_names:
                        txt_file.write(file_name + "\n")

            else:

                inferior, superior, inferior_2, superior_2 = boundary_dict[ct_name]
                start = inferior - 5 if inferior - 5 >= 0 else 0
                ends = superior + 6 if superior + 6 < label_data.shape[1] else label_data.shape[1]
                start_2 = inferior_2 - 5 if inferior_2 - 5 >= 0 else 0
                ends_2 = superior_2 + 6 if superior_2 + 6 < label_data.shape[1] else label_data.shape[1]

                end = label_data.shape[1]
                for w in range(1, end):

                    slice_data = label_data[ :, w, :]
                    data = {"label": slice_data, "subject": ct_name, "slice_number": w, "class": label,"space":label_space}
                    saved_file_name = f"{ct_name}_{w:0>3d}.json"
                    save_path = os.path.join(slice_label_save_path, saved_file_name)

                    if not os.path.exists(save_path):
                        save_json(save_path, data)

                    all_saved_file_names.append(saved_file_name)

                    image_file = os.path.join(ct_image_path, saved_file_name)
                    if not os.path.exists(image_file):
                        if im0_data is None:
                            im0_data = read_cavass_file(im0_file)
                        im0_space = get_voxel_spacing(im0_file)
                        image_data = {"image":im0_data[:, w, :] ,
                                      "subject": ct_name,
                                      "slice_number": w,
                                      "space":im0_space,
                                      "mean": WDS_MEAN,
                                      "std": WDS_STD}
                        save_json(image_file, image_data)
                for ww in range(start, ends):
                    saved_file_name = f"{ct_name}_{ww:0>3d}.json"
                    saved_file_names.append(saved_file_name)
                random.shuffle(all_saved_file_names)
                random.shuffle(saved_file_names)
                ALL_txt_file_path = f"/{label}/ALL_{label}_training_samples_{orientation}.txt"
                txt_file_path = f"/{label}/{label}_training_samples_{orientation}.txt"
                with open(ALL_txt_file_path, "w") as txt_file:
                    for file_name in all_saved_file_names:
                        txt_file.write(file_name + "\n")

                with open(txt_file_path, "w") as txt_file:
                    for file_name in saved_file_names:
                        txt_file.write(file_name + "\n")

