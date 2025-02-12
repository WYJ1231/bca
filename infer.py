import shutil
from pathlib import Path
from typing import List, Tuple

from scipy import ndimage

import csv
import os
import random

import nibabel as nib
import cavasspy.ops as cavass
import numpy as np

np.bool = np.bool_
import torch
from scipy.ndimage import measurements, find_objects

from jhammer.checkpoint import load_checkpoint
from jhammer.config import get_config
from jhammer.io import read_txt_2_list, save_json, save_nii
from transforms import ToType, ZscoreNormalization, AddChannel, ResizeImgAndLab
from skimage.morphology import reconstruction
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from dataset.transforms import GetBoundary
from dataset.get_boundary import get_boundary

from cfgs.args import parser
from dataset.image_dataset import JSONImageDataset
from inference import inference_zoo
from materials.meas_stds_const import WDS_MEAN, WDS_STD
from models import model_zoo
from medpy import metric
from medpy.metric import hd95
from jhammer.metrics import dsc

def preprocess(image, ground_truth, target_size):

    labels, _ = measurements.label(ground_truth)
    objects: List[Tuple[slice]] = list(find_objects(labels))
    volumes = [np.sum(labels[obj] == i + 1) for i, obj in enumerate(objects)]
    max_indices = np.argpartition(volumes, -2)[-2:]
    max_indices = sorted(max_indices, key=lambda i: volumes[i], reverse=True)

    largest_index = max_indices[0] + 1
    second_largest_index = max_indices[1] + 1
    print(f"Largest component index: {largest_index}, volume: {volumes[max_indices[0]]}")
    print(f"Second largest component index: {second_largest_index}, volume: {volumes[max_indices[1]]}")

    mask = np.zeros_like(ground_truth)
    mask[labels == largest_index] = 1
    mask[labels == second_largest_index] = 1

    labels_1, _ = measurements.label(mask)
    slices = find_objects(labels_1)
    print("slices:", slices)
    combined_slice = tuple([
        slice(
            min(slc[dim].start for slc in slices if slc[dim] is not None),
            max(slc[dim].stop for slc in slices if slc[dim] is not None)
        )
        for dim in range(mask.ndim)
    ])
    print("combined_slice:", combined_slice)

    adjustment_slices = []
    for dim in range(mask.ndim):
        current_length = combined_slice[dim].stop - combined_slice[dim].start
        if current_length < target_size[dim]:

            extra = target_size[dim] - current_length
            if (combined_slice[dim].stop + extra // 2 ) > mask.shape[dim] - 1:
                print("11111111111")
                stop = mask.shape[dim]
                start = stop - target_size[dim]
            elif (combined_slice[dim].start - (extra - extra // 2)) < 0 :
                print("2222222")
                start = 0
                stop = target_size[dim]
            else:
                print("3333333")
                start = combined_slice[dim].start - extra // 2
                stop = combined_slice[dim].stop + (extra - extra // 2)

            adjustment_slices.append(slice(start, stop))
        else:

            extra = current_length - target_size[dim]

            adjustment_slices.append(slice(
                combined_slice[dim].start + extra // 2,
                combined_slice[dim].stop - (extra - extra // 2)
            ))

    cropped_image = image[adjustment_slices[0], adjustment_slices[1], adjustment_slices[2]]
    cropped_mask = mask[adjustment_slices[0], adjustment_slices[1], adjustment_slices[2]]

    print(f"Adjusted image region shape: {cropped_image.shape}")
    print(f"Adjusted mask region shape: {cropped_mask.shape}")
    bbox = [[adjustment_slices[0].start,adjustment_slices[0].stop],
            [adjustment_slices[1].start,adjustment_slices[1].stop],
            [adjustment_slices[2].start,adjustment_slices[2].stop]]

    return cropped_image, cropped_mask, bbox


def listdir(path):
    dirs = os.listdir(path)
    dirs.sort()
    return dirs

def euclidean_distance(point1, point2):
    """
    计算两点之间的欧氏距离
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def vote(f_name,voter_paths):
    voter = nib.load(os.path.join(voter_paths[0],f_name))
    merged = voter.get_fdata()
    print("voting {}:".format(f_name))

    for i in range(1, len(voter_paths)):
        voter_f = nib.load(os.path.join(voter_paths[i],f_name))
        merged += voter_f.get_fdata()
    merged[merged > len(voter_paths) / 2] = 1
    merged[merged != 1] =0

    return merged

def hard_voting(voter_paths, f_name):
    voter  = nib.load(os.path.join(voter_paths[0], f_name))
    merged  = voter.get_fdata()
    # print("dtype:", merged.dtype)

    print("Voting {}:".format(f_name))
    for i in range(1, len(voter_paths)):
        voter_f = nib.load(os.path.join(voter_paths[i], f_name))
        voter_f = voter_f.get_fdata()
        merged += voter_f

    # Threshold to determine final prediction
    threshold = len(voter_paths) / 2.0
    merged = np.where(merged >= threshold, 1, 0)

    return merged.astype(int)

def weighted_vote(f_name, voter_paths, weights=None, value = None):
    # 设置第一个面为基准
    voter  = nib.load(os.path.join(voter_paths[0], f_name))
    merged  = voter.get_fdata()
    merged  = merged * weights[0]
    print("weight:", weights)
    print("voting {}:".format(f_name))

    for i in range(1, len(voter_paths)):
        voter_f = nib.load(os.path.join(voter_paths[i], f_name))
        merged += voter_f.get_fdata() * weights [i]

    merged[merged > np.sum(weights) / value] = 1
    merged[merged != 1] = 0

    return merged

def fill_holes(data, label):
    # Fill holes for area label
    area_labels = ["AxR", "LCR"]
    if label not in area_labels:
        return data

    for i in range(data.shape[2]):
        image = data[..., i].astype(bool)
        seed = np.copy(image)
        seed[1:-1, 1:-1] = image.max()
        mask = image
        filled = reconstruction(seed, mask, method="erosion")
        data[..., i] = filled

    return data

def connected_component(prediction):
    labels, _ = measurements.label(prediction)
    objects: List[Tuple[slice]] = list(find_objects(labels))
    volumes = [np.sum(labels[obj] == i + 1) for i, obj in enumerate(objects)]

    if len(objects) > 2:
        max_indices = np.argpartition(volumes, -3)[-3:]
    else:
        max_indices = np.argpartition(volumes, -2)[-2:]
    max_indices = sorted(max_indices, key=lambda i: volumes[i], reverse=True)
    # print("max_indices:", max_indices)
    largest_index = max_indices[0] + 1
    second_largest_index = max_indices[1] + 1
    # print(f"Largest component index: {largest_index}, volume: {volumes[max_indices[0]]}")
    # print(f"Second largest component index: {second_largest_index}, volume: {volumes[max_indices[1]]}")

    mask = np.zeros_like(prediction)
    mask[labels == largest_index] = 1
    mask[labels == second_largest_index] = 1
    if len(objects) > 2:
        third = max_indices[2] + 1
        print(f"third largest component index: {third}, volume: {volumes[max_indices[2]]}")
        mask[labels == third] = 1
        if volumes[max_indices[2]] < 10:
            mask[labels == third] = 0

    return mask

def get_depth_info(ct_name, depth_info_path):
    depth_info = None
    with open(depth_info_path, 'r') as file:
        for line in file:
            if ct_name in line:
                depth_info = int(line.split(': ')[1])
                break
    return depth_info


def main():

    labels = cfg.inference_labels
    target_size = []
    orientations = cfg.orientation
    infer_size = cfg.inference_size
    print("infer_size:", infer_size)

    for label in tqdm(labels):
        if label == 'AxR':
            target_size = cfg.target_size_AxR
        if label == 'LCR':
            target_size = cfg.target_size_LCR
        print("label:",label)
        models = cfg.labels[label].model
        print("models:",models)
        for i, model_n in enumerate(models):
            print("model_n:",model_n)
            if "inference_samples" in cfg:
                if isinstance(cfg.inference_samples, str):
                    samples = read_txt_2_list(cfg.inference_samples.replace("label", label))
                else:
                    samples = cfg.inference_samples
            else:
                samples = [each[:-4] for each in os.listdir(cfg.IM0_path) if each.endswith(".IM0")]

            print("samples:",samples)

            # convert IM0 to json
            if cfg.convert_json:
                print("======Convert IM0 to JSON======")
                for subject_name in tqdm(samples):
                    json_file = os.path.join(cfg.volume_json_path, f"{subject_name}.json")
                    if os.path.exists(json_file):
                        continue
                    im0_file = os.path.join(cfg.IM0_path, f"{subject_name}.IM0")
                    image_data = cavass.read_cavass_file(im0_file)
                    json_obj = {"image": image_data, "subject": subject_name}
                    save_json(json_file, json_obj)
            print("======Infer subjects======")

            nii_save_path_mul_csv = cfg.resulet_nii_path_mul.replace("model", model_n)
            nii_save_path_mul_csv = nii_save_path_mul_csv.replace("label", label)

            # for target_label in tqdm(cfg.inference_labels):
            voter_paths = []
            for orientation in orientations:

                #  test dataset and dataloader
                volume_json_path = cfg.volume_json_path
                volume_label_path = cfg.volume_label_path

                test_samples = [f"{each}.json" for each in samples]
                test_label_dict = {label: volume_label_path.replace("label", label)}

                tr_transforms = Compose([
                    ToType(keys=["image", "imageOriginal"], dtype=np.float32),
                    ZscoreNormalization(keys=["image", "imageOriginal"], mean_value=WDS_MEAN, std_value=WDS_STD),
                    # ResizeImgAndLab(keys=["image", label], target_size=[512, 256], orientation=orientation)
                    # AddChannel(keys=["image"], dim=0)
                    # GetBoundary(keys=["image", label], boundary_dict=boundary_dict, orientation = orientation)
                ])

                dataset = JSONImageDataset(sample_list = test_samples,
                                           sample_dir = volume_json_path,
                                           label_dict = test_label_dict,
                                           target_size= target_size,
                                           transforms=tr_transforms
                                           )
                data_loader = DataLoader(dataset, 1)

                model = model_zoo[cfg.labels[label].model[i]](cfg.labels[label]).to(device)
                checkpoint = cfg.labels[label].checkpoint.replace("model",model_n)
                checkpoint = checkpoint.replace("orientation",orientation)
                load_checkpoint(checkpoint, model)
                inference_method = cfg.labels[label].inference_method if "inference_method" in cfg.labels[label] else cfg.labels[label].model[i]
                print("inference_method:",inference_method)

                inference_method_extra_args = cfg.labels[label].inference_method_args if "inference_method_args" in cfg.labels[label] else {}
                print(inference_zoo.keys())  # 输出所有有效键，确保包含 'vit_base_16'
                inference_list = inference_zoo[inference_method](model, cfg.inference_size, device, **inference_method_extra_args)

                result_path = cfg.result_cavass_path.replace("model", model_n)
                result_path = result_path.replace("label", label)
                nii_save_path_ind = Path(result_path.replace("orientation", orientation))
                if not nii_save_path_ind.exists():
                    nii_save_path_ind.mkdir(parents=True, exist_ok=True)

                voter_paths.append(nii_save_path_ind)
                print("nii_save_path:",nii_save_path_ind)
                print("voter_paths:",voter_paths)
                print("voter_paths[0]:",voter_paths[0])

                csv_file_ind = f"{nii_save_path_mul_csv}/metrics_ind_{orientation}.csv"
                print("单方面的指标存放路径：", nii_save_path_mul_csv)
                print("单方面的指标存放路径csv：", csv_file_ind)

                for batch in tqdm(data_loader):
                    ct_name = batch["subject"][0]
                    ct_im0 = os.path.join(cfg.IM0_path, f"{ct_name}.IM0")
                    space = cavass.get_voxel_spacing(ct_im0)

                    image = batch["image"]
                    image_original = batch["imageOriginal"]
                    bbox = batch["bbox"]
                    label_gt = batch[label]

                    batch['superior_slice'] = None
                    batch['inferior_slice'] = None

                    output_file_path = os.path.join(nii_save_path_ind, f"{ct_name}_{label}.nii")
                    if os.path.exists(output_file_path):
                        continue

                    result = inference_list.infer(batch, orientation)
                    prediction = result.cpu().numpy()

                    # post-process
                    prediction = connected_component(prediction)
                    prediction = fill_holes(prediction, label)
                    # print("prediction_shape:", prediction.shape)

                    inferior_slice = batch["inferior_slice"] if "inferior_slice" in batch else None
                    superior_slice = batch["superior_slice"] if "superior_slice" in batch else None

                    if inferior_slice and superior_slice:
                        original_image_size = image.squeeze().shape
                        output = np.zeros(original_image_size, dtype=np.uint8)
                        output[..., inferior_slice:superior_slice + 1] = prediction[..., inferior_slice:superior_slice + 1]
                    else:
                        # save_bim
                        # output = prediction.astype(np.uint8)
                        # ct_im0 = os.path.join(cfg.IM0_path, f"{ct_name}.IM0")
                        # cavass.save_cavass_file(os.path.join(nii_save_path, f"{ct_name}_{label}.BIM"), output, True,
                        #                         reference_file=ct_im0)

                        output = prediction.astype(np.uint8)
                        label_gt = connected_component(label_gt[0].cpu().numpy())

                        dice = metric.binary.dc(output, label_gt)
                        hd = hd95(output, label_gt, voxelspacing = space)
                        asd = metric.binary.asd(output, label_gt, voxelspacing = space)

                        if not os.path.exists(csv_file_ind):
                            with open(csv_file_ind, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(["Name", "Dice", "Hausdorff Distance", "Average Surface Distance"])

                        with open(csv_file_ind, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([ct_name, dice, hd, asd])

                        # save_nii

                        original_output = np.zeros_like(image_original[0])
                        print("bbox[0][0]", bbox[0][0])
                        print("bbox[0][1]", bbox[0][1])
                        original_output[bbox[0][0]:bbox[0][1],
                                        bbox[1][0]:bbox[1][1],
                                        bbox[2][0]:bbox[2][1]] = output
                        shape = output.shape
                        depth = get_depth_info(ct_name,"/depth_info.txt")

                        # # save_nii
                        # if orientation == "Transverse":
                        #     # new_output = ndimage.zoom(output, zoom=[512/shape[0], 512/shape[1], depth / shape[2]])
                        #     save_nii(os.path.join(nii_save_path_ind, f"{ct_name}_{label}.nii"), original_output, space, "ARI")
                        # else:
                        #     # new_output = ndimage.zoom(output, zoom=[1, 1, depth / shape[2]])
                        #     save_nii(os.path.join(nii_save_path_ind, f"{ct_name}_{label}.nii"), original_output, space, "ARI")

                data = []
                with open(csv_file_ind, mode='r') as file:
                    reader = csv.reader(file)
                    # header = next(reader)
                    for row in reader:
                        data.append(row)

                if len(data) > 1:
                    dice_values = [float(row[1]) for row in data[1:]]
                    hd_values = [float(row[2]) for row in data[1:]]
                    asd_values = [float(row[3]) for row in data[1:]]

                    dice_avg = np.mean(dice_values)
                    hd_avg = np.mean(hd_values)
                    asd_avg = np.mean(asd_values)

                    dice_std_all = np.std(dice_values)
                    hd_std_all = np.std(hd_values)
                    asd_std_all = np.std(asd_values)

                    dice_values_exclude = [value for value in dice_values if
                                   value != max(dice_values) and value != min(dice_values)]
                    hd_values_exclude = [value for value in hd_values if
                                 value != max(hd_values) and value != min(hd_values)]
                    asd_values_exclude = [value for value in asd_values if
                                              value != max(asd_values) and value != min(asd_values)]

                    dice_avg_ind_exclude = np.mean(dice_values_exclude)
                    hd_avg_ind_exclude = np.mean(hd_values_exclude)
                    asd_avg_ind_exclude = np.mean(asd_values_exclude)

                    dice_std_ind_exclude = np.std(dice_values_exclude)
                    hd_std_ind_exclude = np.std(hd_values_exclude)
                    asd_std_ind_exclude = np.std(asd_values_exclude)

                    with open(csv_file_ind, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["mean", dice_avg, hd_avg, asd_avg])
                        writer.writerow(["std", dice_std_all, hd_std_all, asd_std_all])
                        writer.writerow( ["mean (excluding min and max)", dice_avg_ind_exclude, hd_avg_ind_exclude, asd_avg_ind_exclude])
                        writer.writerow( ["std (excluding min and max)", dice_std_ind_exclude, hd_std_ind_exclude, asd_std_ind_exclude])

            # voter_paths = [f"/Cross/{model_n}_mulloss_3/{label}/Transverse", f"/Cross/{model_n}_mulloss_3/{label}/Coronal", f"//Cross/{model_n}_mulloss_3/{label}/Sagittal"]
            # print("voter_paths：", voter_paths)
            # voter_names = [listdir(path) for path in voter_paths]
            # metrics_list = []
            # nii_save_path_mul_csv = f"/Cross/{model_n}_mulloss_3/{label}"
            # csv_file = f"{nii_save_path_mul_csv}/metrics_mul.csv"

            # # for f_name in voter_names[0]:
            # #     # gt
            # #     prefix = f_name.split('_')[0]
            # #     bim_file = os.path.join(f"/{label}", f"{prefix}_{label}.BIM")
            # #     gt_label = cavass.read_cavass_file(bim_file)
            # #     gt_label = connected_component(gt_label)
            # #
            # #     nii_mul = hard_voting(voter_paths, f_name)
            # #     nii_mul = connected_component(nii_mul)

            # #     dice_all = metric.binary.dc(nii_mul, gt_label)
            # #     asd_all = metric.binary.asd(nii_mul, gt_label)
            # #     metrics_list.append([f_name, dice_all, asd_all])
            # #     print("metrics_list:", metrics_list)
            # #
            # #     if not os.path.exists(csv_file):
            # #         with open(csv_file, mode='a', newline='') as file:
            # #             writer = csv.writer(file)
            # #             writer.writerow(["Name", "Dice", "Average Surface Distance"])
            # #
            # #     with open(csv_file, mode='a', newline='') as file:
            # #         writer = csv.writer(file)
            # #         writer.writerow([f_name, dice_all, asd_all])

            # # x = [0.4, 0.3, 0.5, 0.7, 0.8, 0.9, 0.1, 0.2]
            # # y = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]
            # x = [0.4]
            # y = [0.2]
            # value = [1.7]
            # # value = [1.7, 1.6, 1.8, 1.9, 2.0, 2.1, 1.1, 1.2, 1.3, 1.4, 1.5,]
            # ratio = 0
            # dice_values_all, hd_values_all, asd_values_all = [], [], []
            # dice_avg_all, hd_avg_all, asd_avg_all = 0, 0, 0
            # [x_value, y_value, z] = [0, 0, 0]
            # max_weight_dice, min_weight_hd, min_weight_asd = [0, 0, 0], [0, 0, 0], [0, 0, 0]
            # max_dice, min_hd, min_asd = 0, 1000, 1000
            # max_dice_exclude, min_hd_exclude, min_asd_exclude = 0, 0, 0
            #
            # csv_file_exclude = f"{nii_save_path_mul_csv}/metrics_mul_exclude.csv"
            # for v in range(len(value)):
            #     value_value = value[v]
            #     for x_i in range(len(x)):
            #         x_value = x[x_i]
            #         for j in range(len(y)):
            #             y_value = y[j]
            #             if x_value + y_value <= 1:
            #                 z = round(1 - x_value - y_value, 1)
            #                 for f_name in voter_names[0]:
            #                     # gt
            #                     prefix = f_name.split('_')[0]
            #                     bim_file = os.path.join(f"/{label}",f"{prefix}_{label}.BIM")
            #                     gt_label = cavass.read_cavass_file(bim_file)
            #                     gt_label = connected_component(gt_label)

            #                     dsd = []
            #                     for i in range(len(voter_paths)):
            #                         single_pre = nib.load(os.path.join(voter_paths[i], f_name))
            #                         single_pre = single_pre.get_fdata()
            #                         dice = metric.binary.dc(single_pre, gt_label)
            #                         dsd.append(dice)
            #                     sorted_indexes = sorted(range(len(dsd)), key=lambda k: dsd[k], reverse=True)
            #                     with open(f"/Cross/{model_n}_mulloss_3/{label}/dice_mul.txt", "a") as f:
            #                         f.write(f"{f_name}_dice: " + str(dsd) + "\n")

            #                     # nii_mul = weighted_vote(f_name, voter_paths, dsd, sorted_indexes, weights=[x_value, y_value, z], value=value_value)
            #                     nii_mul = weighted_vote(f_name, voter_paths, weights=[x_value, y_value, z], value=value_value)
            #                     nii_mul = connected_component(nii_mul)
            #
            #                     # save_nii
            #                     prefix = f_name.split('_')[0]
            #                     ct_im0 = os.path.join(cfg.IM0_path, f"{prefix}.IM0")
            #                     space = cavass.get_voxel_spacing(ct_im0)
            #                     # save_nii(nii_save_path_mul_csv, nii_mul, space, "ARI")
            #                     save_nii(os.path.join(nii_save_path_mul_csv, 'multi', f"{f_name}"), nii_mul, space, "ARI")

            #                     dice_all = metric.binary.dc(nii_mul, gt_label)
            #                     print("dice_all:", dice_all)
            #                     nii_mul_pre, gt_label_pre, _ = preprocess(nii_mul, gt_label, [256,512,64])
            #                     hd_all = hd95(nii_mul_pre, gt_label_pre)
            #                     print("hd_all:", hd_all)
            #                     asd_all = metric.binary.asd(nii_mul, gt_label)
            #                     metrics_list.append([f_name, dice_all, hd_all, asd_all])
            #                     print("metrics_list:", metrics_list)
            #
            #                     if not os.path.exists(csv_file):
            #                         with open(csv_file, mode='a', newline='') as file:
            #                             writer = csv.writer(file)
            #                             writer.writerow(["Name", "Dice", "HD95","Average Surface Distance"])
            #
            #                     with open(csv_file, mode='a', newline='') as file:
            #                         writer = csv.writer(file)
            #                         writer.writerow([f_name, dice_all, hd_all, asd_all])

            #                 data = []
            #                 for item in metrics_list:
            #                     data.append(item)
            #
            #                 dice_values_all = [float(row[1]) for row in data]
            #                 hd_values_all = [float(row[2]) for row in data]
            #                 asd_values_all = [float(row[3]) for row in data]
            #
            #                 dice_avg_all = np.mean(dice_values_all)
            #                 hd_avg_all = np.mean(hd_values_all)
            #                 asd_avg_all = np.mean(asd_values_all)
            #                 print("dice_avg_all:",dice_avg_all)
            #                 print("hd_avg_all:",hd_avg_all)
            #                 print("asd_avg_all:",asd_avg_all)
            #
            #                 dice_std_all = np.std(dice_values_all)
            #                 hd_std_all = np.std(hd_values_all)
            #                 asd_std_all = np.std(asd_values_all)

            #                 if dice_avg_all > max_dice:
            #                     max_dice = dice_avg_all
            #                     max_weight_dice = [x_value, y_value, z]
            #                     ratio = value[v]
            #                     print("ratio: ", ratio)
            #                     with open(f"/{label}/dice_best.txt", "a") as f:
            #                         f.write("dice更新啦,值为: " + str(max_dice) + "\n")
            #                         f.write("weight: " + str(max_weight_dice) + "\n")
            #                         f.write("ratio: " + str(ratio) + "\n")
            #
            #                 with open(csv_file, mode='a', newline='') as file:
            #                     writer = csv.writer(file)
            #                     writer.writerow(["mean", dice_avg_all, hd_avg_all, asd_avg_all])
            #                     writer.writerow(["std", dice_std_all, hd_std_all, asd_std_all])
            #
            #                 if hd_avg_all < min_hd:
            #                     min_hd = hd_avg_all
            #                     min_weight_hd = [x_value, y_value, z]
            #                     ratio = value[v]
            #                     print("hd更新啦,值为: ", min_hd)
            #                     print("weight: ", min_weight_hd)
            #                     print("ratio: ", ratio)
            #                     with open(f"/Cross/{model_n}_mulloss_3/{label}/hd_best.txt", "a") as f:
            #                         f.write("hd更新啦,值为: " + str(min_hd) + "\n")
            #                         f.write("weight: " + str(min_weight_hd) + "\n")
            #                         f.write("ratio: " + str(ratio) + "\n")
            #
            #                 if asd_avg_all < min_asd:
            #                     min_asd = asd_avg_all
            #                     min_weight_asd = [x_value, y_value, z]
            #                     ratio = value[v]
            #                     print("asd更新啦,值为: ", min_asd)
            #                     print("weight: ", min_weight_asd)
            #                     print("ratio: ", ratio)
            #                     with open(f"/{label}/asd_best.txt", "a") as f:
            #                         f.write("asd更新啦,值为: " + str(min_asd) + "\n")
            #                         f.write("weight: " + str(min_weight_asd) + "\n")
            #                         f.write("ratio: " + str(ratio) + "\n")

            # # for f_name in voter_names[0]:
            # #     nii_mul_asd = weighted_vote(f_name, voter_paths, weights=min_weight_asd, value=ratio)
            # #     nii_mul_asd = nii_mul_asd.astype(np.uint8)
            # #
            # #     nii_mul_hd = weighted_vote(f_name, voter_paths, weights=min_weight_hd, value=ratio)
            # #     nii_mul_hd = nii_mul_hd.astype(np.uint8)
            # #
            # #     nii_mul_dice = weighted_vote(f_name, voter_paths, weights=max_weight_dice, value=ratio)
            # #     nii_mul_dice = nii_mul_dice.astype(np.uint8)
            # #
            # #     prefix = f_name.split('_')[0]
            # #     ct_im0 = os.path.join(cfg.IM0_path, f"{prefix}.IM0")
            # #     space = cavass.get_voxel_spacing(ct_im0)
            # #
            # #
            # #     save_nii(os.path.join(nii_save_path_mul_csv, f"{min_weight_asd}_{ratio}", f"{f_name}_asd_{min_asd}"), nii_mul_asd, space, "ARI")
            # #     save_nii(os.path.join(nii_save_path_mul_csv, f"{min_weight_hd}_{ratio}", f"{f_name}_hd_{min_hd}"), nii_mul_hd, space, "ARI")
            # #     save_nii(os.path.join(nii_save_path_mul_csv, f"{max_weight_dice}_{ratio}", f"{f_name}_dice_{max_dice}"), nii_mul_dice, space, "ARI")
            #
            # with open(csv_file, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(["Average_dice", max_dice, hd_avg_all, asd_avg_all, max_weight_dice, ratio])
            #     writer.writerow(["Average_hd", dice_avg_all, min_hd, asd_avg_all, min_weight_hd, ratio])
            #     writer.writerow(["Average_asd", dice_avg_all, hd_avg_all, min_asd, min_weight_asd, ratio])

            # dice_values_all_exclude = [value for value in dice_values_all if value != max(dice_values_all) and value != min(dice_values_all)]
            # hd_values_all_exclude = [value for value in hd_values_all if value != max(hd_values_all) and value != min(hd_values_all)]
            # asd_values_all_exclude = [value for value in asd_values_all if value != max(asd_values_all) and value != min(asd_values_all)]

            # dice_avg_all_exclude = np.mean(dice_values_all_exclude)
            # hd_avg_all_exclude = np.mean(hd_values_all_exclude)
            # asd_avg_all_exclude = np.mean(asd_values_all_exclude)

            # if dice_avg_all_exclude > max_dice:
            #     print("exclude_dice_更新啦")
            #     max_dice_exclude = dice_avg_all_exclude
            #     max_weight_dice = [x_value, y_value, z]
            #     ratio = ratio
            #
            # if hd_avg_all_exclude < min_hd:
            #     print("exclude_hd_更新啦")
            #     min_hd_exclude = hd_avg_all_exclude
            #     min_weight_hd = [x_value, y_value, z]
            #     ratio = ratio
            #
            # if asd_avg_all_exclude < min_asd:
            #     print("exclude_asd_更新啦")
            #     min_asd_exclude = asd_avg_all_exclude
            #     min_weight_asd = [x_value, y_value, z]
            #     ratio = ratio

            # with open(csv_file_exclude, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(["Average (excluding min and max)_dice", max_dice_exclude, hd_avg_all_exclude, asd_avg_all_exclude, max_weight_dice, ratio])
            #     writer.writerow(["Average (excluding min and max)_hd", dice_avg_all_exclude, min_hd_exclude, asd_avg_all_exclude, min_weight_hd, ratio])
            #     writer.writerow(["Average (excluding min and max)_asd", dice_avg_all_exclude, hd_avg_all_exclude, min_asd_exclude, min_weight_asd, ratio])
            #
            # # save_nii
            # # for f_name in voter_names[0]:
            # #     nii_mul_asd_exclude = weighted_vote(f_name, voter_paths, weights=min_weight_asd, value=ratio)
            # #     nii_mul_asd_exclude = nii_mul_asd_exclude.astype(np.uint8)
            # #
            # #     nii_mul_hd_exclude = weighted_vote(f_name, voter_paths, weights=min_weight_hd, value=ratio)
            # #     nii_mul_hd_exclude = nii_mul_hd_exclude.astype(np.uint8)
            # #
            # #     nii_mul_dice_exclude = weighted_vote(f_name, voter_paths, weights=max_weight_dice, value=ratio)
            # #     nii_mul_dice_exclude = nii_mul_dice_exclude.astype(np.uint8)
            # #
            # #     prefix = f_name.split('_')[0]
            # #     ct_im0 = os.path.join(cfg.IM0_path, f"{prefix}.IM0")
            # #     space = cavass.get_voxel_spacing(ct_im0)
            # #
            # #     # save_nii
            # #     save_nii(os.path.join(nii_save_path_mul_csv, f"{min_weight_asd}_{ratio}",f"{f_name}_asd_{min_asd_exclude}"), nii_mul_asd_exclude, space, "ARI")
            # #     save_nii(os.path.join(nii_save_path_mul_csv, f"{min_weight_hd}_{ratio}", f"{f_name}_hd_{min_hd_exclude}"), nii_mul_hd_exclude, space, "ARI")
            # #   save_nii(os.path.join(nii_save_path_mul_csv, f"{max_weight_dice}_{ratio}", f"{f_name}_dice_{max_dice_exclude}"), nii_mul_dice_exclude, space, "ARI")





if __name__ == "__main__":
    args = parser.parse_args()
    cfg = get_config(args.cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cfg.gpus)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main()

    # file_path = '/volume/volume_AxR_json/N017PETCT.json'

    # with open(file_path, 'r') as file:
    #     data = json.load(file)

    # print(data)
