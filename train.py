import os
import re
from random import shuffle
import torch.nn.functional as F
import numpy as np
import torch
# import monai
from einops import rearrange
# from medpy import metric
from pandas import read_csv

from jhammer.checkpoint import load_checkpoint, save_checkpoint
from jhammer.config import get_config
from jhammer.io import read_txt_2_list
from jhammer.losses import DiceLoss
from jhammer.lr_schedulers import poly_lr
from jhammer.lr_utils import get_lr
from jhammer.metrics import dsc
# from jhammer.models import UNet, UNet3d, UnetCross, UNetPlusPlus, VisionTransformer, C2BAMUNet, CrossSliceAttentionUNet
from jhammer.models.unet import CrossSliceAttentionUNet, C2BAMUNet
from jhammer.samplers import GridSampler
from transforms import ZscoreNormalization, ToType, AddChannel, ResizeImgAndLab
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm, trange

from cfgs.args import parser
from dataset.image_dataset import JSONImageDataset
from materials.meas_stds_const import WDS_MEAN, WDS_STD

# 关于数据集报错
def main():

    def to_patch(input_batch, label, orientation):
        # BHWD
        image = input_batch["image"]  # image: torch.Size([B, 512, 512, D])
        gt_label = input_batch[label] # gt: torch.Size([B, 512, 512, D])
        batch_size = cfg.batch_size

        if orientation == "Transverse":
            image = rearrange(image, "b h w d -> (b d) h w")
            gt_label = rearrange(gt_label, "b h w d -> (b d) h w")
            batch_size = cfg.batch_size

        if orientation == "Coronal":
            image = rearrange(image, "b h w d -> (b h) w d")
            gt_label = rearrange(gt_label, "b h w d -> (b h) w d")
            print("Coronal  image:", image.size())
            print("Coronal  gt_label:", gt_label.size())

        if orientation == "Sagittal":
            image = rearrange(image, "b h w d -> (b w) h d")
            gt_label = rearrange(gt_label, "b h w d -> (b w) h d")

        batch_size = batch_size if batch_size < image.shape[1] else image.shape[1]
        patch_size = (batch_size, image.size(1), image.size(2))
        image_sampler = GridSampler(image, patch_size)
        label_sampler = GridSampler(gt_label, patch_size)

        return image_sampler, label_sampler

    # # 2D: 直接训练切片
    # def train_batch(input_batch, label):
    #     # BCHW # HWD
    #     image = input_batch["image"] # image: torch.Size([26, 1, 512, 512])
    #     next_image = input_batch["next_image"]
    #     pre_image = input_batch["pre_image"]
    #     # 在通道维度上连接三个图片
    #     concatenated_images = torch.cat((pre_image, image, next_image), dim=1)
    #     # print("concatenated_images:",concatenated_images.shape)  # 输出 torch.Size([26, 3, 512, 512])
    #
    #     gt_label = input_batch[label] # gt_label: torch.Size([26, 512, 512])
    #     output = model(concatenated_images.to(device)) # output: torch.Size([26, 2, 512, 512])
    #     # bce_loss = F.cross_entropy(output, gt_label.long().to(device))
    #     dice_loss = dice_loss_fun(output, gt_label.to(device))
    #     # loss = bce_loss + dice_loss
    #     return dice_loss    # def train_batch(input_batch, label):
    #     #     # BCHW # HWD
    #     #     image = input_batch["image"] # image: torch.Size([26, 1, 512, 512])
    #     #     next_image = input_batch["next_image"]
    #     #     pre_image = input_batch["pre_image"]
    #     #     # 在通道维度上连接三个图片
    #     #     concatenated_images = torch.cat((pre_image, image, next_image), dim=1)
    #     #     # print("concatenated_images:",concatenated_images.shape)  # 输出 torch.Size([26, 3, 512, 512])
    #     #
    #     #     gt_label = input_batch[label] # gt_label: torch.Size([26, 512, 512])
    #     #     output = model(concatenated_images.to(device)) # output: torch.Size([26, 2, 512, 512])
    #     #     # bce_loss = F.cross_entropy(output, gt_label.long().to(device))
    #     #     dice_loss = dice_loss_fun(output, gt_label.to(device))
    #     #     # loss = bce_loss + dice_loss
    #     #     return dice_loss

    # 3D_input: 1slice   NO.1
    def train_batch(image_batch, label_batch):
            # image_batch: torch.Size([4, 512, 512])
            # slice_output_patch: torch.Size([1, 2, 512, 512, 4])
            slice_size = cfg.batch_size
            slice_size = slice_size if slice_size < image_batch.shape[1] else image_batch.shape[1]
            slice_patch_size = (slice_size, image_batch.size(1), image_batch.size(2))
            slice_image_sampler = GridSampler(image_batch, slice_patch_size)

            slice_output = []
            ori_slice_output = []
            for image_slice in slice_image_sampler:
                image_slice = image_slice.unsqueeze(dim=1)  # [b/d, 1, H, W]
                # output, x, coor_info
                # slice_output_patch, ori_output_patch,_ = model(image_slice.to(device))  # [b/d, 2, H, W]
                slice_output_patch = model(image_slice.to(device))  # [b/d, 2, H, W]
                slice_output_patch = rearrange(slice_output_patch, "d c h w -> 1 c h w d")
                # ori_output_patch = rearrange(ori_output_patch, "d c h w -> 1 c h w d")
                # print("ori_output_patch:", ori_output_patch.size())  #  torch.Size([1, 2, 512, 512, B/D])
                # print("slice_output_patch:",slice_output_patch.size())  #  torch.Size([1, 2, 512, 512, B/D])
                slice_output.append(slice_output_patch)
                # ori_slice_output.append(ori_output_patch)
            slice_output = torch.cat(slice_output, dim = -1)
            # ori_slice_output = torch.cat(ori_slice_output, dim = -1)
            # print("slice_output:", slice_output.size())  # slice_output: torch.Size([1, 2, 512, 512, 12])
            label_batch = rearrange(label_batch, "d h w -> 1 h w d")

            bce_loss = F.cross_entropy(slice_output, label_batch.long().to(device))
            # bce_loss_ori = F.cross_entropy(ori_slice_output, label_batch.long().to(device))
            patch_loss = dice_loss_fun(slice_output, label_batch.to(device))
            # patch_loss_ori = dice_loss_fun(ori_slice_output, label_batch.to(device))
            # print("patch_loss:", patch_loss.item()) # patch_loss: 0.3337397277355194
            # total_loss = bce_loss + patch_loss + bce_loss_ori + patch_loss_ori
            total_loss = bce_loss + patch_loss

            return total_loss

    # 3D_input: val_volume_1slice  NO.1
    @torch.no_grad()
    def infer_3d_volume(val_batch, label, orientation):
        val_image = val_batch["image"]  # val_image: torch.Size([1, 512, 512, 443])
        if orientation == "Transverse":
            val_image = rearrange(val_image, "b h w d -> (b d) h w")
        elif orientation == "Coronal":
            val_image = rearrange(val_image, "b h w d -> (b h) w d")
            print("Coronal_val_shape:", val_image.shape)
        else:
            val_image = rearrange(val_image, "b h w d -> (b w) h d")
            print("Sagittal_val_shape:", val_image.shape)

        batch_size = cfg.val_batch_size
        batch_size = batch_size if batch_size < val_image.shape[1] else val_image.shape[1]
        patch_size = (batch_size, val_image.size(1), val_image.size(2))
        sampler = GridSampler(val_image, patch_size)

        output = []
        for patch in sampler:
            patch = patch.unsqueeze(dim=1)
            # output, x, coor_info
            # output_patch,_,_ = model(patch.to(device))  # tensor(1.0000, device='cuda:0')
            output_patch = model(patch.to(device))  # tensor(1.0000, device='cuda:0')
            output_patch = torch.argmax(output_patch, dim=1).to(torch.uint8).cpu()  # tensor(1, dtype=torch.uint8) output_patch: torch.Size([B/d, 512, 512])
            # print("output_patch:", output_patch.size())
            output.append(output_patch)
        output = sampler.restore(output)  # val_result: torch.Size([585, 512, 512])
        return output

    # # 3张切片作为channel   NO.2
    # def train_batch(image_batch, label_batch):
    #     # print("image_batch:",image_batch.size())  # image_batch: torch.Size([B, 512, 512])
    #     # print("label_batch:",label_batch.size())  # label_batch: torch.Size([B, 512, 512])
    #     slice_size = image_batch.shape[0]
    #     slice_output = []
    #     slice_output_ori = []
    #     adjacent_slices = []
    #     for i in range(slice_size):
    #         current_slice = image_batch[i]
    #         pre_slice = image_batch[i -1] if i > 0 else image_batch[i]
    #         next_slice = image_batch[i + 1] if i < (slice_size - 1) else image_batch[i]
    #         mul_slice = torch.stack([pre_slice, current_slice, next_slice], dim = 0) # mul_slice: torch.Size([3, 512, 512])
    #         # print("mul_slice:", mul_slice.size()) # torch.Size([C = 3, 512, 512])
    #         adjacent_slices.append(mul_slice)
    #     adjacent_slices = torch.stack(adjacent_slices)
    #     # print("adjacent_slices:", adjacent_slices.size()) # adjacent_slices: torch.Size([B, 3, 512, 512])
    #     # slice_output_patch = model(adjacent_slices.to(device))
    #     slice_output_patch, slice_output_patch_ori, _= model(adjacent_slices.to(device))
    #     # print("slice_output_patch_0:", slice_output_patch.size()) # slice_output_patch_0: torch.Size([B, 2, 512, 512])
    #     slice_output_patch = rearrange(slice_output_patch, "d c h w -> 1 c h w d")
    #     slice_output_patch_ori = rearrange(slice_output_patch_ori, "d c h w -> 1 c h w d")
    #     # print("slice_output_patch_rearrange:",slice_output_patch.size()) # slice_output_patch_rearrange: torch.Size([1, 2, 512, 512, B/D])
    #     slice_output.append(slice_output_patch)
    #     slice_output_ori.append(slice_output_patch_ori)
    #     slice_output = torch.cat(slice_output, dim = -1)
    #     slice_output_ori = torch.cat(slice_output_ori, dim = -1)
    #     # print("slice_output_end:", slice_output.size())  # slice_output_end: torch.Size([1, 2, 512, 512, B/D])
    #     label_batch = rearrange(label_batch, "d h w -> 1 h w d")
    # 
    #     bce_loss = F.cross_entropy(slice_output, label_batch.long().to(device))
    #     patch_loss = dice_loss_fun(slice_output, label_batch.to(device))
    #     # total_loss = bce_loss + patch_loss
    #     bce_loss_ori = F.cross_entropy(slice_output_ori, label_batch.long().to(device))
    #     patch_loss_ori = dice_loss_fun(slice_output_ori, label_batch.to(device))
    #     total_loss = bce_loss + patch_loss + bce_loss_ori + patch_loss_ori
    #     # print("patch_loss:", patch_loss.item()) # patch_loss: 0.3337397277355194
    #     return total_loss
    # 
    # #  val_volume_3slice NO.2
    # @torch.no_grad()
    # def infer_3d_volume(val_batch, label, orientation):
    #     val_image = val_batch["image"]  # val_image: torch.Size([1, 512, 512, 443])
    #     print("val_image：", val_image.size())
    #     val_batch_size = cfg.val_batch_size
    # 
    #     if orientation == "Transverse":
    #         val_image = rearrange(val_image, "b h w d -> (b d) h w")
    #         # if label == "AxR":
    #         #     val_batch_size = val_batch_size - 6
    # 
    #     elif orientation == "Coronal":
    #         val_image = rearrange(val_image, "b h w d -> (b h) w d")
    # 
    #     elif orientation == "Sagittal":
    #         val_image = rearrange(val_image, "b h w d -> (b w) h d")
    #         # print("val_image_Sagittal:", val_image.size())
    #         # if label == "AxR":
    #         #     val_batch_size = val_batch_size * 2
    #         # else:
    #         #     val_batch_size = val_batch_size * 4
    # 
    #     val_batch_size = val_batch_size if val_batch_size < val_image.shape[1] else val_image.shape[1]
    #     patch_size = (val_batch_size, val_image.size(1), val_image.size(2))
    #     sampler = GridSampler(val_image, patch_size)
    # 
    #     output = []
    #     for patch in sampler:
    #         adjacent_slices = []
    #         for i in range(patch.size(0)):
    #             # print("val_patch:", patch.size())  # torch.Size([B, 512, 512])
    #             current_slice = patch[i]
    #             # print("current_slice:",current_slice.size()) # current_slice: torch.Size([512, 512])
    #             pre_slice = patch[i - 1] if i > 0 else patch[i]
    #             # print("val_image_rearrange:", val_image.size())  #  torch.Size([234, 512, 512])
    #             next_slice = patch[i + 1] if i < (patch.size(0) - 1) else patch[i]
    #             mul_slice = torch.stack([pre_slice, current_slice, next_slice],dim=0)
    #             # print("val_mul_slice:", mul_slice.size()) # val_mul_slice: torch.Size([C, 512, 512])
    #             adjacent_slices.append(mul_slice)
    #         patch = torch.stack(adjacent_slices)  # patch: torch.Size([b, 3, 512, 512])
    #         # print("val_adjacent_slices:", patch.size()) # val_adjacent_slices: torch.Size([16, 3, 512, 512])
    #         # patch = patch.unsqueeze(dim=1) # 3 1 512 512
    #         output_patch, _, _ = model(patch.to(device))  # tensor(1.0000, device='cuda:0')
    #         # output_patch = model(patch.to(device))  # tensor(1.0000, device='cuda:0')
    #         output_patch = torch.argmax(output_patch, dim=1).to(torch.uint8).cpu()  # tensor(1, dtype=torch.uint8) output_patch: torch.Size([B/d, 512, 512])
    #         print("val_output_patch:", output_patch.size())
    #         output.append(output_patch)
    # 
    #     output = sampler.restore(output)  # val_result: torch.Size([585, 512, 512])
    #     return output

    # # 3d UNet 对比
    # def train_batch(image_batch, label_batch):
    #
    #     # image_batch: torch.Size([4, 512, 512])
    #     # slice_output_patch: torch.Size([1, 2, 512, 512, 4])
    #
    #     label_batch = rearrange(label_batch, "d h w -> 1 h w d")
    #     batch_size = cfg.batch_size
    #     batch_size = batch_size if batch_size < image_batch.shape[1] else image_batch.shape[1]
    #     patch_size = (batch_size, image_batch.size(1), image_batch.size(2))
    #     image_sampler = GridSampler(image_batch, patch_size)
    #
    #     total_loss = 0
    #     for patch_image in image_sampler:
    #         patch_image = patch_image.unsqueeze(dim=1)  # [b/d, 1, H, W]
    #         patch_image = rearrange(patch_image, "d c h w -> 1 c d h w")
    #         # print("patch_image:", patch_image.size()) # patch_image: torch.Size([1, 1, 16, 512, 512])
    #         output_patch = model(patch_image.to(device))  # [1, 2, D, H, W]
    #         # print("output_patch:", output_patch.size()) # output_patch: torch.Size([1, 2, 16, 512, 512])
    #         output_patch = rearrange(output_patch, "b c d h w -> b c h w d")
    #
    #         bce_loss = F.cross_entropy(output_patch, label_batch.long().to(device))
    #         patch_loss = dice_loss_fun(output_patch, label_batch.to(device))
    #         # print("patch_loss:", patch_loss.item()) # patch_loss: 0.3337397277355194
    #         total_loss = bce_loss + patch_loss
    #
    #     return total_loss
    #
    # # 3D UNet
    # @torch.no_grad()
    # def infer_3d_volume(val_batch, label, orientation):
    #     val_image = val_batch["image"]  # val_image: torch.Size([1, 512, 512, 443])
    #     if orientation == "Transverse":
    #         val_image = rearrange(val_image, "b h w d -> (b d) h w")
    #     elif orientation == "Coronal":
    #         val_image = rearrange(val_image, "b h w d -> (b h) w d")
    #         print("Coronal_val_shape:", val_image.shape)
    #     else:
    #         val_image = rearrange(val_image, "b h w d -> (b w) h d")
    #         print("Sagittal_val_shape:", val_image.shape)
    #
    #     batch_size = cfg.val_batch_size
    #     batch_size = batch_size if batch_size < val_image.shape[1] else val_image.shape[1]
    #     patch_size = (batch_size, val_image.size(1), val_image.size(2))
    #     sampler = GridSampler(val_image, patch_size)
    #
    #     output = []
    #     for patch in sampler:
    #         patch = patch.unsqueeze(dim=1) # d c h w
    #         patch = rearrange(patch, "d c h w -> 1 c d h w")
    #         output_patch = model(patch.to(device))  # 1 2 d h w
    #         output_patch = torch.argmax(output_patch, dim=1).to(torch.uint8).cpu()  # output_patch: torch.Size([1,B/d, 512, 512])
    #         print("output_patch:", output_patch.size())
    #         output.append(output_patch)
    #     output = sampler.restore(output)  # val_result: torch.Size([585, 512, 512])
    #     return output


    labels = cfg.label
    orientations = cfg.orientation
    task = cfg.task
    in_channels = cfg.in_channels
    target_size = None

    for label in labels:
        for orientation in orientations:
            if label == 'AxR':
                target_size = cfg.target_size_AxR
            if label == 'LCR':
                target_size = cfg.target_size_LCR
            # # vit
            # model = VisionTransformer(cfg, img_size = target_size, num_classes = cfg.n_classes).to(device)
            # dice_loss_fun = DiceLoss(to_one_hot_y=True)
            # optimizer = SGD(params=model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-3)
            # grad_scaler = GradScaler()

            # Model
            # model = UNetPlusPlus(in_channels, cfg.n_classes).to(device)
            # model = UNet(in_channels=in_channels, out_channels=cfg.n_classes).to(device)
            # model = UNet3d(in_channels=in_channels, out_channels=cfg.n_classes).to(device)
            # model = UnetCross(in_channels, cfg.n_classes).to(device)
            model = C2BAMUNet(in_channels, cfg.n_classes, num_layers=5, batch_size = cfg.batch_size).to(device)
            # model = CrossSliceAttentionUNet(in_channels, cfg.n_classes, num_layers=5, batch_size = cfg.batch_size).to(device)

            dice_loss_fun = DiceLoss(to_one_hot_y=True)
            optimizer = SGD(params=model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-3)
            grad_scaler = GradScaler()

            print(f"----------{label}---{orientation}---------")

            # # 2D training dataset and data loader
            # if orientation =='Transverse':
            #     tr_transforms = Compose([
            #         ToType(keys=["image", "next_image", "pre_image"], dtype=np.float32),
            #         ZscoreNormalization(keys=["image", "next_image", "pre_image"], mean_value=WDS_MEAN, std_value=WDS_STD, mean_key="mean", std_key="std"),
            #         AddChannel(keys=["image", "next_image", "pre_image"], dim=0)
            #     ])
            # else:
            #     tr_transforms = Compose([
            #         ToType(keys=["image", "next_image", "pre_image"], dtype=np.float32),
            #         ZscoreNormalization(keys=["image", "next_image", "pre_image"], mean_value=WDS_MEAN, std_value=WDS_STD, mean_key="mean", std_key="std"),
            #         AddChannel(keys=["image", "next_image", "pre_image"], dim=0)
            #         ,ResizeImgAndLab(keys = ["image", "next_image", "pre_image",label],target_size = [512,256], orientation = orientation)
            #     ])
            #
            # # 直接读取2D切片
            # training_sample_txt = cfg.training_sample_txt.replace("label", label)
            # training_sample_txt = training_sample_txt.replace("orientation",orientation)
            # training_samples = read_txt_2_list(training_sample_txt)
            #
            # training_image_dir = cfg.slice_sample_dir.image.replace("orientation", orientation)
            # training_label_dict = {label: cfg.slice_sample_dir[label].replace("orientation", orientation)}
            # training_dataset = JSONImageDataset(phase = phase,
            #                                     sample_list=training_samples,
            #                                     sample_dir=training_image_dir,
            #                                     label_dict=training_label_dict,
            #                                     transforms=tr_transforms)
            #
            # training_data_loader = DataLoader(dataset=training_dataset,
            #                                   batch_size=cfg.batch_size,
            #                                   shuffle= True,
            #                                   num_workers=2,
            #                                   pin_memory=True)

            # 3D training dataset and data loader
            tr_transforms = Compose([
                        ToType(keys="image", dtype=np.float32),
                        ZscoreNormalization(keys="image", mean_value=WDS_MEAN, std_value=WDS_STD, mean_key="mean", std_key="std"),
                        # ResizeImgAndLab(keys=["image", label], orientation=orientation, target_size= target_size)
                        # AddChannel(keys=["image", "next_image", "pre_image"], dim=0)
                    ])

            train_sample_txt = cfg.training_sample_txt.replace("label", label)
            train_samples = read_txt_2_list(train_sample_txt)
            volume_image_dir = cfg.volume_sample_dir.image
            train_label_dict = {label: cfg.volume_sample_dir[label]}

            train_dataset = JSONImageDataset(sample_list=train_samples,
                                             sample_dir=volume_image_dir,
                                             target_size=target_size,
                                             label_dict=train_label_dict,
                                             phase='train',
                                             add_postfix=True,
                                             transforms=tr_transforms)
            training_data_loader = DataLoader(train_dataset,
                                           shuffle= True,
                                           batch_size = 5)

            # val dataset and data loader
            # boundary_file = cfg.boundary_file.replace("label", label)
            # boundary_file = boundary_file.replace("orientation",orientation)
            # boundary_dict = get_boundary(boundary_file, orientation)
            val_transforms = Compose([
                ToType(keys="image", dtype=np.float32),
                ZscoreNormalization(keys="image", mean_value=WDS_MEAN, std_value=WDS_STD, mean_key="mean", std_key="std"),
                # ResizeImgAndLab(keys=["image", label], orientation=orientation, target_size= target_size)
                # GetBoundary(keys=["image", label], boundary_dict=boundary_dict, orientation = orientation)
            ])

            val_sample_txt = cfg.val_sample_txt.replace("label", label)
            val_samples = read_txt_2_list(val_sample_txt)
            # volume_sample_dir
            volume_image_dir = cfg.volume_sample_dir.image
            val_label_dict = {label: cfg.volume_sample_dir[label]}

            val_dataset = JSONImageDataset(sample_list=val_samples,
                                           sample_dir=volume_image_dir,
                                           target_size=target_size,
                                           label_dict=val_label_dict,
                                           phase='val',
                                           add_postfix=True,
                                           transforms=val_transforms)
            val_data_loader = DataLoader(val_dataset, 1)

            best_val_dice = 0
            second_best_val_dice = 0
            third_best_val_dice = 0
            fourth_best_val_dice = 0
            fifth_best_val_dice = 0
            iteration = 0
            snapshot_dir = os.path.join(cfg.snapshot,f"bca{task}",f"{label}", f"{orientation}")
            os.makedirs(snapshot_dir, exist_ok=True)
            checkpoint_dir = os.path.join(snapshot_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            vis_log_path = os.path.join(snapshot_dir, 'log')
            vis_log = SummaryWriter(vis_log_path)

            # Load checkpoint
            if "checkpoint" in cfg and cfg.checkpoint:
                load_checkpoint(cfg.checkpoint, model)

            # Train
            for epoch in trange(0, cfg.epochs):
                print("-"*15)
                poly_lr(optimizer=optimizer, initial_lr=cfg.lr, epoch=epoch, num_epochs=cfg.epochs)
                vis_log.add_scalar("lr", get_lr(optimizer), epoch)
                model.train()
                for batch in tqdm(training_data_loader):
                    iteration += 1
                    sampler_image, sampler_label = to_patch(batch, label, orientation)
                    for image_patch, label_patch in zip(sampler_image, sampler_label):
                        optimizer.zero_grad()
                        # print("patch:", image_patch.size())  # patch: torch.Size([12, 512, 512])
                        # print("gt_patch:", label_patch.size())  # gt_patch: torch.Size([12, 512, 512])
                        with autocast():
                            loss = train_batch(image_patch, label_patch)
                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        vis_log.add_scalar('loss', loss.detach().cpu(), iteration)

                # validate epoch
                val_interval = cfg.val_interval if "val_interval" in cfg else 1
                if epoch != 0 and epoch % val_interval == 0:
                    model.eval()
                    val_dice_scores = []
                    for val_batch in tqdm(val_data_loader):
                        val_result = infer_3d_volume(val_batch, label, orientation)
                        print("val_result:", val_result.shape)
                        if orientation == "Transverse":
                            val_result = rearrange(val_result, "d h w -> 1 h w d")
                        elif orientation == "Coronal":
                            val_result = rearrange(val_result, "h w d -> 1 h w d")
                        elif orientation == "Sagittal":
                            val_result = rearrange(val_result, "w h d -> 1 h w d")

                        target = val_batch[label]
                        val_dice_score = dsc(val_result, target)
                        val_dice_scores.append(val_dice_score)
                    val_dice_score = torch.stack(val_dice_scores).mean()

                    vis_log.add_scalar('val dice', val_dice_score, epoch)
                    if val_dice_score >= best_val_dice:
                        # 最优解
                        second_best_val_dice = best_val_dice
                        best_val_dice = val_dice_score
                        save_checkpoint(os.path.join(snapshot_dir, f'{label}_unet_{orientation}.pth'), model)

                        vis_log.add_scalar('snapshot/epoch', epoch, epoch)
                        vis_log.add_scalar('snapshot/dice', best_val_dice, epoch)

                    elif val_dice_score >= second_best_val_dice:
                        # 次优解
                        third_best_val_dice = second_best_val_dice
                        second_best_val_dice = val_dice_score
                        save_checkpoint(os.path.join(snapshot_dir, f'{label}_unet_{orientation}_second_best.pth'), model)
                        vis_log.add_scalar('snapshot/epoch_second_best', epoch, epoch)
                        vis_log.add_scalar('snapshot/dice_second_best', second_best_val_dice, epoch)

                    elif val_dice_score >= third_best_val_dice:
                        # 次次优解
                        fourth_best_val_dice = third_best_val_dice
                        third_best_val_dice = val_dice_score
                        save_checkpoint(os.path.join(snapshot_dir, f'{label}_unet_{orientation}_third_best.pth'), model)
                        vis_log.add_scalar('snapshot/epoch_third_best', epoch, epoch)
                        vis_log.add_scalar('snapshot/dice_third_best', third_best_val_dice, epoch)

                    elif val_dice_score >= fourth_best_val_dice:
                        fifth_best_val_dice = fourth_best_val_dice
                        fourth_best_val_dice = val_dice_score
                        save_checkpoint(os.path.join(snapshot_dir, f'{label}_unet_{orientation}_fourth_best.pth'), model)
                        vis_log.add_scalar('snapshot/epoch_fourth_best', epoch, epoch)
                        vis_log.add_scalar('snapshot/dice_fourth_best', fourth_best_val_dice, epoch)

                    elif val_dice_score >= fifth_best_val_dice:
                        fifth_best_val_dice = val_dice_score
                        save_checkpoint(os.path.join(snapshot_dir, f'{label}_unet_{orientation}_fifth_best.pth'), model)
                        vis_log.add_scalar('snapshot/epoch_fifth_best', epoch, epoch)
                        vis_log.add_scalar('snapshot/dice_fifth_best', fifth_best_val_dice, epoch)


                    dice_file = os.path.join(snapshot_dir, f"{orientation}{task}_Dice.txt")
                    with open(dice_file, 'a') as file:
                        file.write(f'Epoch {epoch} - Validation Dice Score: {val_dice_score}\n')
                dice_file_0 = os.path.join(snapshot_dir, f"{orientation}{task}_Best_Dice.txt")
                with open(dice_file_0, 'a') as file:
                    file.write(f'Best Value: {best_val_dice}\n')


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args.cfg)
    print("------当前配置文件---------")
    print(" task:",cfg.task)
    print(" epoch:",cfg.epochs)
    print(" batch_size:",cfg.batch_size)
    # print(" num_per_image_AxR:",cfg.num_per_image_AxR)
    # print(" num_per_image_LCR:",cfg.num_per_image_LCR)
    print(" val_batch_size:",cfg.val_batch_size)
    model = cfg.model
    in_channels = cfg.in_channels
    print(" model:", cfg["model"][0])
    print(" in_channels:",cfg.in_channels)
    if args.gpus:
        cfg.gpus = args.gpus
    device = 'cpu'
    if torch.cuda.is_available() and 'gpus' in cfg:
        device = torch.device('cuda')
        if 'cudnn' in cfg:
            cudnn.benchmark = cfg.cudnn.benchmark
            cudnn.deterministic = cfg.cudnn.deterministic
    pid = os.getpid()
    print("Current_pid:",pid)
    main()

