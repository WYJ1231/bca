import copy
from collections import OrderedDict
from functools import reduce

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td

from torch.autograd import Function

# import correlation_cuda
#
# # ######################################################################### UnetCross #############################################################################################################################

# class CorrelationFunction(Function):
#
#     def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
#         super(CorrelationFunction, self).__init__()
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply
#         # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)
#
#     @staticmethod
#     def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
#         ctx.input1 = input1
#         ctx.input2 = input2
#         ctx.pad_size = pad_size
#         ctx.kernel_size = kernel_size
#         ctx.max_displacement = max_displacement
#         ctx.stride1 = stride1
#         ctx.stride2 = stride2
#         ctx.corr_multiply = corr_multiply
#
#         # ctx.save_for_backward(self, input1, input2)

#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()
#             output = input1.new()

#             correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
#                                      pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
#
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input1 = ctx.input1
#         input2 = ctx.input2
#         pad_size = ctx.pad_size
#         kernel_size = ctx.kernel_size
#         max_displacement = ctx.max_displacement
#         stride1 = ctx.stride1
#         stride2 = ctx.stride2
#         corr_multiply = ctx.corr_multiply
#
#         # self, input1, input2 = ctx.saved_tensors
#
#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()
#
#             grad_input1 = input1.new()
#             grad_input2 = input2.new()
#
#             correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
#                                       pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
#
#         del (ctx.input1)
#         del (ctx.input2)
#         del (ctx.pad_size)
#         del (ctx.kernel_size)
#         del (ctx.max_displacement)
#         del (ctx.stride1)
#         del (ctx.stride2)
#         del (ctx.corr_multiply)
#
#         return grad_input1, grad_input2, None, None, None, None, None, None
#
# class Correlation(nn.Module):
#     def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
#         super(Correlation, self).__init__()
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply
#
#     def forward(self, input1, input2):
#         result = CorrelationFunction.apply(input1, input2,
#                                            self.pad_size,
#                                            self.kernel_size,
#                                            self.max_displacement,
#                                            self.stride1,
#                                            self.stride2,
#                                            self.corr_multiply)
#
#         return result
#
# # embedding
# def local_pairwise_distances(model, x, y):
#     """Computes pairwise squared l2 distances using a local search window.
#
#     Optimized implementation using correlation_cost.
#
#     Args:
#     x: Float32 tensor of shape [batch, feature_dim, height, width].
#     y: Float32 tensor of shape [batch, feature_dim, height, width].
#     max_distance: Integer, the maximum distance in pixel coordinates
#       per dimension which is considered to be in the search window.
#
#     Returns:
#     Float32 distances tensor of shape
#       [batch, (2 * max_distance + 1) ** 2, height, width].
#     """
#     # d[i,j] = (x[i] - y[j]) * (x[i] - y[j])'
#     # = sum(x[i]^2, -1) + sum(y[j]^2, -1) - 2 * x[i] * y[j]'
#
#     corr = model.corr(x, y)
#     xs = torch.sum(x * x, dim=1, keepdim=True)
#     ys = torch.sum(y * y, dim=1, keepdim=True)
#     ones_ys = torch.ones_like(ys)
#     ys = model.corr(ones_ys, ys)
#     d = xs.half() + ys - 2 * corr     # feature dist

#     # add div
#     d = d / d.shape[1]
#
#     # Boundary should be set to Inf.
#     boundary = torch.eq(model.corr(ones_ys, ones_ys), 0)
#
#     # d = torch.where(boundary, torch.ones_like(d).fill_(np.float('inf')), d)
#     d = torch.where(boundary, torch.ones_like(d).fill_(float('inf')), d)
#
#     return d
#
# def local_previous_frame_nearest_neighbor_features_per_object(model,
#         prev_frame_embedding, query_embedding, prev_frame_labels,
#         max_distance=9, save_cost=True, device=None):
#     """Computes nearest neighbor features while only allowing local matches.
#
#     Args:
#       prev_frame_embedding: Tensor of shape [batch, embedding_dim, height, width],
#         the embedding vectors for the last frame.
#       query_embedding: Tensor of shape [batch, embedding_dim, height, width],
#         the embedding vectors for the query frames.
#       prev_frame_labels: Tensor of shape [batch, 1, height, width], the class labels of
#         the previous frame.
#       gt_ids: Int Tensor of shape [n_objs] of the sorted unique ground truth
#         ids in the first frame.
#       max_distance: Integer, the maximum distance allowed for local matching.
#
#     Returns:
#       nn_features: A float32 np.array of nearest neighbor features of shape
#         [batch, (2*d+1)**2, height,width].
#     """
#
#     assert device is not None, "Device should not be none."
#
#     #         query_embedding,
#     #         prev_frame_embedding,
#     if save_cost:
#         d = local_pairwise_distances(model, query_embedding, prev_frame_embedding)  # shape:(batch, (2*d+1)**2, height, width)
#     else:
#         # Slow fallback in case correlation_cost is not available.
#         pass
#     d = (torch.sigmoid(d) - 0.5) * 2
#     batch = prev_frame_embedding.size()[0]
#     height = prev_frame_embedding.size()[-2]
#     width = prev_frame_embedding.size()[-1]
#
#     # Create offset versions of the mask.
#     if save_cost:
#         # New, faster code with cross-correlation via correlation_cost.
#         # Due to padding we have to add 1 to the labels.
#         # offset_labels = models.corr4(torch.ones((1, 1, height, width)).to(device), torch.unsqueeze((prev_frame_labels.permute((2, 0, 1)) + 1).to(device), 0))
#
#         offset_labels = model.corr(torch.ones((batch, 1, height, width)).to(device), (prev_frame_labels + 1).to(device))
#         # offset_labels = offset_labels.permute((2, 3, 1, 0))
#         # Subtract the 1 again and round.
#         offset_labels = torch.round(offset_labels - 1)
#         offset_masks = torch.eq(offset_labels, 1).type(torch.uint8)
#         # shape:(batch,(2*d+1)**2, height,width)
#     else:
#         # Slower code, without dependency to correlation_cost
#         pass

#     pad = torch.ones((batch, (2 * max_distance + 1) ** 2, height, width)).type(torch.half).to(device)
#     # shape:(batch, (2*d+1)**2, height,width)
#     d_tiled = d.half()  # shape:(batch, (2*d+1)**2, height,width)
#     d_masked = torch.where(offset_masks, d_tiled, pad)

#     dists = torch.min(d_masked, dim=1, keepdim=True)[0].float()  # shape:(batch, 1, height,width)
#     return dists
#
# def get_logits_with_matching(model,
#                              features,
#                              reference_labels,
#                              ref_len,
#                              correlation_window_size,
#                              save_cost):
#     # height = features.size(2)
#     # width = features.size(3)
#
#
#     embedding = model.embedding_conv2d(features)
#     label = reference_labels

#     ref_embedding = embedding.clone()
#     ref_embedding[ref_len:] = embedding[:-ref_len]
#     ref_embedding[:ref_len] = embedding[ref_len: 2 * ref_len]
#
#     ref_label = label.clone()
#     ref_label[ref_len:] = label[:-ref_len]
#     ref_label[:ref_len] = label[ref_len: 2 * ref_len]
#
#     '''for debug'''
#     # embedding = embedding[:4, 0, 0, 0]
#     # print(embedding)
#     # ref_embedding = embedding.clone()     # [-0.6149, -0.5467, -0.6859, -0.6736]
#     # ref_embedding[ref_len:] = embedding[:-ref_len]
#     # ref_embedding[:ref_len] = embedding[ref_len: 2 * ref_len]
#     # print(ref_embedding)                  # [-0.5467, -0.6149, -0.5467, -0.6859]
#
#     # 局部相关性，它在当前嵌入和参考嵌入之间基于提供的最大距离（correlation_window_size）进行了一定的相关性
#     coor_info = local_previous_frame_nearest_neighbor_features_per_object(
#         model,
#         ref_embedding,
#         embedding,
#         ref_label,
#         max_distance=correlation_window_size,
#         save_cost=save_cost,
#         device=model.device)
#
#
#     features_n_concat = torch.cat([features, label, coor_info], dim=1)
#     out_embedding = model.embedding_seg_conv2d(features_n_concat)
#     return out_embedding, coor_info
#
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             # nn.Linear(channel, channel // reduction, bias=False),
#             nn.Linear(channel, channel // reduction),
#             nn.ReLU(inplace=True),
#             # nn.Linear(channel // reduction, channel, bias=False),
#             nn.Linear(channel // reduction, channel),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
#
#
# class STB(nn.Module):
#     def __init__(self, device, inplane, outplane, ref_len, correlation_window_size, save_cost, embedding_dim):
#         super(STB, self).__init__()
#         self.device = device
#         self.inplane = inplane
#
#         inplane = reduce(lambda x, y: x + y, self.inplane)
#         self.feat_conv = SingleConv(inplane, outplane, 1, 1, 0)
#         self.ref_len = ref_len
#         self.correlation_window_size = correlation_window_size
#         self.save_cost = save_cost
#         self.embedding_dim = embedding_dim
#
#         self.embedding_conv2d = DoubleConvCross(in_channels_1=outplane, out_channels_1=embedding_dim,
#                                            kernel_size_1=[3, 3], stride_1=(1, 1), padding_1=[1, 1],
#                                            in_channels_2=embedding_dim, out_channels_2=embedding_dim,
#                                            kernel_size_2=[3, 3], stride_2=(1, 1), padding_2=[1, 1])
#
#         self.embedding_seg_conv2d = DoubleConvCross(in_channels_1=outplane+1+1, out_channels_1=embedding_dim,
#                                            kernel_size_1=[3, 3], stride_1=(1, 1), padding_1=[1, 1],
#                                            in_channels_2=embedding_dim, out_channels_2=embedding_dim,
#                                            kernel_size_2=[3, 3], stride_2=(1, 1), padding_2=[1, 1])
#         if save_cost:
#             self.corr = Correlation(pad_size=correlation_window_size, kernel_size=1,
#                                     max_displacement=correlation_window_size, stride1=1,
#                                     stride2=1, corr_multiply=embedding_dim)     # wait to check
#
#         self.se = SELayer(inplane + embedding_dim)
#         self.merge_conv2d = nn.Sequential(OrderedDict([
#                 ('conv', nn.Conv2d(inplane + embedding_dim, embedding_dim, 3, 1, 1)),
#                 # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
#                 ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
#                 ('instnorm',
#                  nn.InstanceNorm2d(embedding_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
#                 ('conv_out', nn.Conv2d(embedding_dim, 2, 3, 1, 1)),
#             ]))
#
#
#     def forward(self, feature, out):
#         # s = out.size()[0]  # batch size
#         # h = out.size()[2]  # 128
#         # w = out.size()[3]  # 128
#
#         high_h = feature[0].size(-2)  # 16
#         high_w = feature[0].size(-1)  # 16
#         feature = reduce(lambda x, y: torch.cat(    # concat low_features
#             (x, F.interpolate(y, size=(high_h, high_w), mode='bilinear', align_corners=True)), dim=1), feature)

#         matching_input = self.feat_conv(feature.detach()) # shape: (batch_size, outplane(256), 128, 128)

#         out = torch.argmax(out, dim=1).unsqueeze(1).float()  # original prediction # shape: (batch_size, 1, 128, 128)

#         stb_embedding, coor_info = get_logits_with_matching(self,
#                                                  matching_input,
#                                                  reference_labels=out.detach(),
#                                                  ref_len=self.ref_len,
#                                                  correlation_window_size=self.correlation_window_size,
#                                                  save_cost=self.save_cost)
#
#         m_input = self.se(torch.cat((stb_embedding, feature), dim=1))  # in_channels: 496 + 128 = 624
#         m_output = self.merge_conv2d(m_input)
#
#         m_output = torch.softmax(m_output, dim=1)

#         return m_output, coor_info
#
#
# class SingleConv(nn.Module):
#     def __init__(self, in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1):
#         super(SingleConv, self).__init__()
#         self.blocks = nn.ModuleList([
#             nn.Sequential(OrderedDict([
#                 ('conv', nn.Conv2d(in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1)),
#                 # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
#                 ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
#                 ('instnorm',
#                  nn.InstanceNorm2d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
#             ]))
#         ])
#
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return x
#
# # class DoubleConvCross(nn.Module):
# #     def __init__(self, in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1,
# #                  in_channels_2, out_channels_2, kernel_size_2, stride_2, padding_2):
# #         super(DoubleConvCross, self).__init__()
# #         self.blocks = nn.ModuleList([
# #             nn.Sequential(OrderedDict([
# #                 ('conv', nn.Conv2d(in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1)),
# #                 ('bn1',nn.BatchNorm2d(out_channels_1)),
# #                 ('rule1', nn.ReLU(inplace=True))
# #             ])),
# #             nn.Sequential(OrderedDict([
# #                 ('conv', nn.Conv2d(in_channels_2, out_channels_2, kernel_size_2, stride_2, padding_2)),
# #                 ('bn2', nn.BatchNorm2d(out_channels_1)),
# #                 ('rule2', nn.ReLU( inplace=True))
# #             ]))
# #         ])
# #
# #     def forward(self, x):
# #         for block in self.blocks:
# #             x = block(x)
# #         return x
#
# class DoubleConvCross(nn.Module):
#     def __init__(self, in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1,
#                  in_channels_2, out_channels_2, kernel_size_2, stride_2, padding_2):
#         super(DoubleConvCross, self).__init__()
#         self.blocks = nn.ModuleList([
#             nn.Sequential(OrderedDict([
#                 ('conv', nn.Conv2d(in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1)),
#                 # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
#                 ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
#                 ('instnorm',
#                  nn.InstanceNorm2d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
#             ])),
#             nn.Sequential(OrderedDict([
#                 ('conv', nn.Conv2d(in_channels_2, out_channels_2, kernel_size_2, stride_2, padding_2)),
#                 # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
#                 ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
#                 ('instnorm',
#                  nn.InstanceNorm2d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
#             ]))
#         ])
#
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return x
#
#
# '''Vanilla Deep Sup Unet for size [128, 128]'''
# class UnetCross(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UnetCross, self).__init__()
#         self.conv_blocks_context = nn.ModuleList([
#             DoubleConvCross(in_channels_1=in_channels, out_channels_1=16, kernel_size_1=[3, 3], stride_1=(1, 1),
#                        padding_1=[1, 1],
#                        in_channels_2=16, out_channels_2=16, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 16, 128, 128]
#             DoubleConvCross(in_channels_1=16, out_channels_1=32, kernel_size_1=[3, 3], stride_1=[2, 2],
#                        padding_1=[1, 1],
#                        in_channels_2=32, out_channels_2=32, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 32, 64, 64]
#             DoubleConvCross(in_channels_1=32, out_channels_1=64, kernel_size_1=[3, 3], stride_1=[2, 2],
#                        padding_1=[1, 1],
#                        in_channels_2=64, out_channels_2=64, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 64, 32, 32]
#             DoubleConvCross(in_channels_1=64, out_channels_1=128, kernel_size_1=[3, 3], stride_1=[2, 2],
#                        padding_1=[1, 1],
#                        in_channels_2=128, out_channels_2=128, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 128, 16, 16]
#             DoubleConvCross(in_channels_1=128, out_channels_1=256, kernel_size_1=[3, 3], stride_1=[2, 2],
#                        padding_1=[1, 1],
#                        in_channels_2=256, out_channels_2=256, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 256, 8, 8]
#         ])
#
#         self.conv_blocks_localization = nn.ModuleList([
#             DoubleConvCross(in_channels_1=256, out_channels_1=128, kernel_size_1=[3, 3], stride_1=(1, 1),
#                        padding_1=[1, 1],
#                        in_channels_2=128, out_channels_2=128, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 128, 16, 16]
#             DoubleConvCross(in_channels_1=128, out_channels_1=64, kernel_size_1=[3, 3], stride_1=(1, 1),
#                        padding_1=[1, 1],
#                        in_channels_2=64, out_channels_2=64, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 64, 32, 32]
#             DoubleConvCross(in_channels_1=64, out_channels_1=32, kernel_size_1=[3, 3], stride_1=(1, 1),
#                        padding_1=[1, 1],
#                        in_channels_2=32, out_channels_2=32, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 32, 64, 64]
#             DoubleConvCross(in_channels_1=32, out_channels_1=16, kernel_size_1=[3, 3], stride_1=(1, 1),
#                        padding_1=[1, 1],
#                        in_channels_2=16, out_channels_2=16, kernel_size_2=[3, 3], stride_2=(1, 1),
#                        padding_2=[1, 1]),    # [bs, 16, 128, 128]
#         ])
#
#         self.tu = nn.ModuleList([
#             # nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2), bias=False),
#             # nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2), bias=False),
#             # nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2), bias=False),
#             # nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2), bias=False)
#             nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
#             nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
#             nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
#             nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
#             ])
#
#         # self.seg = nn.Conv2d(16, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         self.seg = nn.Conv2d(16, out_channels, kernel_size=(1, 1), stride=(1, 1))
#
#         self.stb = STB(device="cuda",
#                        inplane=[16, 32, 64, 128, 256],  # feature channel
#                        outplane=256,
#                        ref_len=1,
#                        correlation_window_size=3,
#                        save_cost=True,
#                        embedding_dim=128)
#
#     def forward(self, x):
#         x1 = self.conv_blocks_context[0](x)
#         x2 = self.conv_blocks_context[1](x1)
#         x3 = self.conv_blocks_context[2](x2)
#         x4 = self.conv_blocks_context[3](x3)
#         x5 = self.conv_blocks_context[4](x4)
#
#         x = self.tu[0](x5)
#         x = torch.cat((x, x4), dim=1)
#         x = self.conv_blocks_localization[0](x)
#
#         x = self.tu[1](x)
#         x = torch.cat((x, x3), dim=1)
#         x = self.conv_blocks_localization[1](x)
#
#         x = self.tu[2](x)
#         x = torch.cat((x, x2), dim=1)
#         x = self.conv_blocks_localization[2](x)
#
#         x = self.tu[3](x)
#         x = torch.cat((x, x1), dim=1)
#         x = self.conv_blocks_localization[3](x)
#
#         x = self.seg(x)
#
#         # 添加了softmax操作
#         x = torch.softmax(x, dim=1)
#
#         # return x
#
#         mer_output, coor_info = self.stb([x1, x2, x3, x4, x5], x)
#
#         # 最终输出、原始预测结果、距离图(?)
#         return mer_output, x, coor_info

# ############################################################################# 2d  #####################################################################################################
#
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        bn1 = nn.BatchNorm2d(num_features=mid_channels)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1)
        bn2 = nn.BatchNorm2d(num_features=out_channels)
        relu2 = nn.ReLU(inplace=True)
        super().__init__(conv1, bn1, relu1, conv2, bn2, relu2)
# class UNetEncoder(nn.Module):
#     def __init__(self, in_channels=1, width_factor=16, blocks=5):
#         super().__init__()
#         channels = [width_factor << i for i in range(blocks)]
#
#         block_0 = DoubleConv(in_channels=in_channels, out_channels=channels[0])
#         self.blocks = nn.ModuleList([
#             nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels=channels[i - 1], out_channels=channels[i]))
#             for i in range(1, blocks)])
#
#         self.blocks.insert(0, block_0)
#         self.out_channels = channels
#
#     def forward(self, x):
#         features = []
#         for block in self.blocks:
#             x = block(x)
#             features.append(x)
#         return features
# class UNetDecoder(nn.Module):
#     def __init__(self, encoder_channels):
#         super().__init__()
#
#         in_channels = encoder_channels[::-1]
#
#         self.blocks = nn.ModuleList([
#             Up(in_channels=in_channels[i], out_channels=in_channels[i + 1],
#                skip_conn_channels=in_channels[i + 1])
#             for i in range(0, len(in_channels) - 1)])
#
#         self.out_channels = in_channels[-1]
#
#     def forward(self, x):
#         skip_connections = x[-2::-1]
#         x = x[-1]
#         for i, skip_connection in enumerate(skip_connections):
#             x = self.blocks[i](x, skip_connection)
#         return x
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, skip_conn_channels):
#         # print("in_channel:",in_channels)
#         # print("out_channels:",out_channels)
#         # print("skip_conn_channels:",skip_conn_channels)
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         self.conv = DoubleConv(in_channels // 2 + skip_conn_channels, out_channels)
#
#     def forward(self, x, skip_features):
#         x = self.up(x)
#         diff_y = skip_features.size()[2] - x.size()[2]
#         diff_x = skip_features.size()[3] - x.size()[3]
#         if diff_y != 0 or diff_x != 0:
#             x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
#                           diff_y // 2, diff_y - diff_y // 2])
#
#         # print("skip_features:",skip_features.size())
#         x = torch.cat([skip_features, x], dim=1)
#
#         return self.conv(x)
#
# class UNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, width_factor=32, blocks=5, normalize=True):
#         """
#         UNet.
#
#         Args:
#             in_channels (int, optional, default=1):
#             out_channels (int, optional, default=1):
#             width_factor (int, optional, default=64):
#             blocks (int, optional, default=5):
#             normalize (bool, optional, default=True): If `True`, normalize the output using `torch.sigmoid` for one
#                 dimension output, or `torch.softmax` for multiple classes output.
#         """
#
#         super().__init__()
#         self.out_channels = out_channels
#         encoder = UNetEncoder(in_channels=in_channels, width_factor=width_factor, blocks=blocks)
#         decoder = UNetDecoder(encoder_channels=encoder.out_channels)
#         decoder_head = nn.Conv2d(in_channels=decoder.out_channels, out_channels=out_channels, kernel_size=1)
#         self.unet = nn.Sequential(encoder,
#                                   decoder,
#                                   decoder_head)
#         self.normalize = normalize
#
#     def forward(self, x):
#         x = self.unet(x)
#         if self.normalize:
#             if self.out_channels == 1:
#                 x = torch.sigmoid(x)
#             else:
#                 x = torch.softmax(x, dim=1)
#         return x

# # ######################################################################################## 3d ################################################################################################
# class DoubleConv3d(nn.Sequential):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#
#         conv1 = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
#         bn1 = nn.BatchNorm3d(num_features=mid_channels)
#         relu1 = nn.ReLU(inplace=True)
#         conv2 = nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1)
#         bn2 = nn.BatchNorm3d(num_features=out_channels)
#         relu2 = nn.ReLU(inplace=True)
#         super().__init__(conv1, bn1, relu1, conv2, bn2, relu2)
#
# class Up3d(nn.Module):
#     def __init__(self, in_channels, out_channels, skip_conn_channels):
#         # print("in_channel:",in_channels)
#         # print("out_channels:",out_channels)
#         # print("skip_conn_channels:",skip_conn_channels)
#         super().__init__()
#         self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         self.conv = DoubleConv3d(in_channels // 2 + skip_conn_channels, out_channels)
#
#     def forward(self, x, skip_features):
#         x = self.up(x)
#         # print("x_up:", x.size())
#         # print("skip_features[2]:", skip_features.size()[2])
#         # print("x.size()[2]:", x.size()[2])
#         diff_z = skip_features.size()[2] - x.size()[2]
#         if diff_z != 0:
#             # print("---------------------------------------------------------")
#             x = F.pad(x, [0, 0, 0, 0, diff_z // 2, diff_z - diff_z // 2])
#             # print("x.pad.size():", x.size())
#         # print("skip_features:",skip_features.size())
#         x = torch.cat([skip_features, x], dim=1)
#
#         return self.conv(x)
# class UNetEncoder3d(nn.Module):
#     def __init__(self, in_channels=1, width_factor=64, blocks=5):
#         super().__init__()
#         channels = [width_factor << i for i in range(blocks)]
#
#         block_0 = DoubleConv3d(in_channels=in_channels, out_channels=channels[0])
#         self.blocks = nn.ModuleList([
#             # #             nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels=channels[i - 1], out_channels=channels[i]))
#             nn.Sequential(nn.MaxPool3d(3, stride=2, padding=1), DoubleConv3d(in_channels=channels[i - 1], out_channels=channels[i]))
#             for i in range(1, blocks)])
#
#         self.blocks.insert(0, block_0)
#         self.out_channels = channels
#
#     def forward(self, x):
#         features = []
#         for block in self.blocks:
#             x = block(x)
#             features.append(x)
#         # for fe in features:
#         #     print("features:", fe.shape)
#         return features
#
# class UNetDecoder3d(nn.Module):
#     def __init__(self, encoder_channels):
#         super().__init__()
#
#         in_channels = encoder_channels[::-1]
#
#         self.blocks = nn.ModuleList([
#             Up3d(in_channels=in_channels[i], out_channels=in_channels[i + 1],
#                skip_conn_channels=in_channels[i + 1])
#             for i in range(0, len(in_channels) - 1)])
#
#         self.out_channels = in_channels[-1]
#
#     def forward(self, x):
#         skip_connections = x[-2::-1]
#         # for it in skip_connections:
#             # print("skip_connections:", it.shape)
#         x = x[-1]
#         # print("x: ", x.size())
#         for i, skip_connection in enumerate(skip_connections):
#             x = self.blocks[i](x, skip_connection)
#         return x
#
# class UNet3d(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, width_factor=64, blocks=5, normalize=True):
#         """
#         UNet.
#
#         Args:
#             in_channels (int, optional, default=1):
#             out_channels (int, optional, default=1):
#             width_factor (int, optional, default=64):
#             blocks (int, optional, default=5):
#             normalize (bool, optional, default=True): If `True`, normalize the output using `torch.sigmoid` for one
#                 dimension output, or `torch.softmax` for multiple classes output.
#         """
#
#         super().__init__()
#         self.out_channels = out_channels
#         encoder = UNetEncoder3d(in_channels=in_channels, width_factor=width_factor, blocks=blocks)
#         decoder = UNetDecoder3d(encoder_channels=encoder.out_channels)
#         decoder_head = nn.Conv3d(in_channels=decoder.out_channels, out_channels=out_channels, kernel_size=1)
#         self.unet = nn.Sequential(encoder,
#                                   decoder,
#                                   decoder_head)
#         self.normalize = normalize
#
#     def forward(self, x):
#         x = self.unet(x)
#         if self.normalize:
#             if self.out_channels == 1:
#                 x = torch.sigmoid(x)
#             else:
#                 x = torch.softmax(x, dim=1)
#         return x
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  vision tranformer  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        # print("residual:", residual.size())
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            # print("downsample:", residual.size())
            residual = self.gn_proj(residual)
            # print("gn_proj:",residual.size())

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        # print("conv1(x):", self.conv1(x).size())
        # print("gn1(self.conv1(x)):", self.gn1(self.conv1(x)).size())
        y = self.relu(self.gn2(self.conv2(y)))
        # print("conv2(y):", self.conv2(y).size())
        # print("gn2(self.conv2(y)):", self.gn2(self.conv2(y)).size())

        # print("conv3(y):", self.conv3(y).size())
        y = self.gn3(self.conv3(y))
        # print("gn3:", y.size())

        # print("residual + y:", (residual + y).size())
        y = self.relu(residual + y)
        return y

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            # block_units=[3, 4, 9]
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        # print("b:",b)
        # print("c:", c)
        # print("in_size:",in_size)

        x = self.root(x)
        # print("root(x):",x.size())
        features.append(x)
        # for idx, feat in enumerate(features):
        #     print(f"Feature {idx} type: {feat.size()}")
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        # print("Maxpool:",x.size())
        for i in range(len(self.body)-1):
            # print("i:", i)
            x = self.body[i](x)
            # print(f"body{[i]}(x):", x.size())
            right_size = int(in_size / 4 / (i+1))
            # print("right_size:",right_size)
            # print("x.size()[2]:",x.size()[2])
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                # print("pad:", pad)
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                # print("feat:", feat.size())
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        # print("len(feature):",len(features))
        x = self.body[-1](x)
        # print("resnet_x:", x.size())
        # for idx, feat in enumerate(features[::-1]):
        #     print(f"Feature[::-1] {idx} type: {feat.size()}")

        return x, features[::-1]

# Non-Local Block for multi-cross attention
class NLBlockND_multicross_block(nn.Module):
    """
    Non-Local Block for multi-cross attention.

    Args:
        in_channels (int): Number of input channels.
        inter_channels (int, optional): Number of intermediate channels. Defaults to None.

    Attributes:
        in_channels (int): Number of input channels.
        inter_channels (int): Number of intermediate channels.
        g (nn.Conv2d): Convolutional layer for the 'g' branch.
        final (nn.Conv2d): Final convolutional layer.
        W_z (nn.Sequential): Sequential block containing a convolutional layer followed by batch normalization for weight 'z'.
        theta (nn.Conv2d): Convolutional layer for the 'theta' branch.
        phi (nn.Conv2d): Convolutional layer for the 'phi' branch.

    Methods:
        forward(x_thisBranch, x_otherBranch): Forward pass of the non-local block.

    """

    def __init__(self, in_channels, inter_channels=None):
        super(NLBlockND_multicross_block, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.final = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels)
        )

        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

    def forward(self, x_thisBranch, x_otherBranch):
        batch_size = x_thisBranch.size(0)
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        z = self.W_z(y)
        return z


    # Multi-Cross Attention Block
class NLBlockND_multicross(nn.Module):

    def __init__(self, in_channels, inter_channels=None):
        super(NLBlockND_multicross, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.cross_attention = NLBlockND_multicross_block(in_channels=1024, inter_channels=64)

    def forward(self, x_thisBranch, x_otherBranch):
        outputs = []
        for i in range(16):
            cross_attention = NLBlockND_multicross_block(in_channels=1024, inter_channels=64)
            cross_attention = cross_attention.to('cuda')
            output = cross_attention(x_thisBranch, x_otherBranch)

            outputs.append(output)
        final_output = torch.cat(outputs, dim=1)
        # final_output = final_output + x_thisBranch #Changed
        return final_output

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        # print("x_cat:", x.size())
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        # decoder_channels = config.decoder_channels
        decoder_channels = [256, 128, 64, 16]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            # skip_channels = self.config.skip_channels
            skip_channels = [512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        #         print("in_channels:", in_channels)
        #         print("out_channels:", out_channels)
        #         print("skip_channels:", skip_channels)
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # print("hidden_states:", hidden_states.size())
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        # print("x_permute:", x.size())
        x = x.contiguous().view(B, hidden, h, w)
        # print("xxxx:", x.size())
        x = self.conv_more(x)
        # print("conv_more:", x.size())
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class DownCross(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.hybrid_prev = None
        self.hybrid_next = None
        self.config = config
        img_size = nn.modules.utils._pair(img_size)

        self.cross_attention_multi_1 = NLBlockND_multicross(in_channels=1024, inter_channels=512)
        self.cross_attention_multi_2 = NLBlockND_multicross(in_channels=1024, inter_channels=512)
        self.cross_attention_multi_3 = NLBlockND_multicross(in_channels=1024, inter_channels=512)
        self.downcross_three = (DownCross(3072, 1024))

        if config.patches.get("grid") is not None:  # ResNet50
            grid_size = config.patches["grid"]
            # print("grid_size:", grid_size)
            # print("img_size:",img_size)
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            # print("patch_size:", patch_size)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            # print("patch_size_real:", patch_size_real)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            # print("n_patches:", n_patches)
            self.hybrid = True
        else:
            patch_size = nn.modules.utils._pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
            self.hybrid_next = False
            self.hybrid_prev = False
        if self.hybrid:
            # self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            self.hybrid_model = ResNetV2(block_units=[3, 4, 9], width_factor=config.resnet.width_factor)
            self.hybrid_model_prev = ResNetV2(block_units=[3, 4, 9], width_factor=config.resnet.width_factor)
            self.hybrid_model_next = ResNetV2(block_units=[3, 4, 9], width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x_prev, x, x_next):
        if self.hybrid:

            x, features = self.hybrid_model(x)
            x_prev, features1 = self.hybrid_model(x_prev)
            x_next, features2 = self.hybrid_model(x_next)
        else:
            features = None

        xt1 = self.cross_attention_multi_1(x, x_next)
        xt2 = self.cross_attention_multi_2(x, x_prev)
        xt3 = self.cross_attention_multi_3(x, x)

        xt = torch.cat([xt1, xt3, xt2], dim=1)
        x = self.downcross_three(xt)
        # print("downcross(xt):", x.size())

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # print("patch_embeddings(x):", x.size())
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        # print("embedding:",embeddings.size())
        embeddings = self.dropout(embeddings)
        # print("dropout_embedding:", embeddings.size())
        return embeddings, features

# Attention module definition
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        # print("Attention_x:", x.size())
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        # print("MLP_x:", x.size())
        x = x + h
        # print("x+h:", x.size())
        return x, weights

class Encoder(nn.Module):
    # vis = None
    # hidden_size=1
    # transformer["num_layers"]=1
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            # print("hidden_states_encoder_1:", hidden_states.size())
            hidden_states, weights = layer_block(hidden_states)
            # print("hidden_states_encoder_2:", hidden_states.size())
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        # print("encoded:", encoded.size())
        return encoded, attn_weights

# Transformer architecture
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, x_prev,x,x_next):
        embedding_output, features = self.embeddings(x_prev,x,x_next)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features
#
# class Transformer(nn.Module):
#     def __init__(self, config, img_size, vis):
#         super(Transformer, self).__init__()
#         self.embeddings = Embeddings(config, img_size=img_size)
#         self.encoder = Encoder(config, vis)
#
#     def forward(self, x_prev,x,x_next):
#         embedding_output, features = self.embeddings(x_prev,x,x_next)
#         encoded, attn_weights = self.encoder(embedding_output)
#         return encoded, attn_weights, features

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    # def forward(self, x_prev,x,x_next):
    def forward(self, x):
        # x.size()  [B, 3, H, W]
        x_prev = x[:, 0, :, :]
        x_cur = x[:, 1, :, :]
        x_next = x[:, 2, :, :]
        # print("x_prev:", x_prev.size())
        # print("x_cur:", x_cur.size())
        # print("x_next:", x_next.size())

        x_prev = x_prev.unsqueeze(dim=1)
        x_cur = x_cur.unsqueeze(dim=1)
        x_next = x_next.unsqueeze(dim=1)
        # print("x_prev_1:", x_prev.size())
        # print("x_cur_1:", x_cur.size())
        # print("x_next_1:", x_next.size())

        if x_cur.size()[1] == 1:
            x_cur = x_cur.repeat(1,3,1,1)
            x_prev = x_prev.repeat(1,3,1,1)
            x_next = x_next.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x_prev, x_cur, x_next)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

# # #################################################################################################  CSAM-Net  ######################################################################################################################
# import torch
# import torch.nn as nn
# import torch.distributions as td
# import torch.nn.functional as F
# import math
# import numpy as np
#
# def custom_max(x,dim,keepdim=True):
#     temp_x=x
#     for i in dim:
#         temp_x=torch.max(temp_x,dim=i,keepdim=True)[0]
#     if not keepdim:
#         temp_x=temp_x.squeeze()
#     return temp_x
#
# class PositionalAttentionModule(nn.Module):
#     def __init__(self):
#         super(PositionalAttentionModule,self).__init__()
#         self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(7,7),padding=3)
#     def forward(self,x):
#         max_x=custom_max(x,dim=(0,1),keepdim=True)
#         avg_x=torch.mean(x,dim=(0,1),keepdim=True)
#         att=torch.cat((max_x,avg_x),dim=1)
#         att=self.conv(att)
#         att=torch.sigmoid(att)
#         return x*att
#
# class SemanticAttentionModule(nn.Module):
#     def __init__(self,in_features,reduction_rate=16):
#         super(SemanticAttentionModule,self).__init__()
#         self.linear=[]
#         self.linear.append(nn.Linear(in_features=in_features,out_features=in_features//reduction_rate))
#         self.linear.append(nn.ReLU())
#         self.linear.append(nn.Linear(in_features=in_features//reduction_rate,out_features=in_features))
#         self.linear=nn.Sequential(*self.linear)
#     def forward(self,x):
#         max_x=custom_max(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
#         avg_x=torch.mean(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
#         max_x=self.linear(max_x)
#         avg_x=self.linear(avg_x)
#         att=max_x+avg_x
#         att=torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
#         return x*att
#
# class SliceAttentionModule(nn.Module):
#     def __init__(self,in_features,rate=4,uncertainty=True,rank=5):
#         super(SliceAttentionModule,self).__init__()
#         self.uncertainty=uncertainty
#         self.rank=rank
#         self.linear=[]
#         self.linear.append(nn.Linear(in_features=in_features,out_features=int(in_features*rate)))
#         self.linear.append(nn.ReLU())
#         self.linear.append(nn.Linear(in_features=int(in_features*rate),out_features=in_features))
#         self.linear=nn.Sequential(*self.linear)
#         if uncertainty:
#             self.non_linear=nn.ReLU()
#             self.mean=nn.Linear(in_features=in_features,out_features=in_features)
#             self.log_diag=nn.Linear(in_features=in_features,out_features=in_features)
#             self.factor=nn.Linear(in_features=in_features,out_features=in_features*rank)
#     def forward(self,x):
#         max_x=custom_max(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
#         avg_x=torch.mean(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
#         max_x=self.linear(max_x)
#         avg_x=self.linear(avg_x)
#         att=max_x+avg_x
#         if self.uncertainty:
#             temp=self.non_linear(att)
#             mean=self.mean(temp)
#             diag=self.log_diag(temp).exp()
#             factor=self.factor(temp)
#
#             # 将相关张量转换为 float32
#             mean = mean.to(torch.float32)
#             diag = diag.to(torch.float32)
#
#             factor=factor.view(1,-1,self.rank)
#             factor = factor.to(torch.float32)
#
#             print(f'mean dtype: {mean.dtype}, device: {mean.device}')
#             print(f'diag dtype: {diag.dtype}, device: {diag.device}')
#             print(f'factor dtype: {factor.dtype}, device: {factor.device}')
#
#             dist=td.LowRankMultivariateNormal(loc=mean,cov_factor=factor,cov_diag=diag)
#             # # 计算协方差矩阵
#             # cov = torch.matmul(factor, factor.transpose(-1, -2)) + diag  # Cholesky分解的协方差
#             # # 使用Cholesky分解得到协方差矩阵的下三角矩阵
#             # chol_cov = torch.linalg.cholesky(cov)
#             # # 通过低秩多变量正态分布进行采样
#             # dist = td.LowRankMultivariateNormal(loc=mean, cov_factor=chol_cov)
#
#             print(f'att dtype: {att.dtype}, device: {att.device}')
#             att=dist.sample().to(torch.float32)
#         att=torch.sigmoid(att).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         return x*att
#
#
# class CSAM(nn.Module):
#     def __init__(self,num_slices,num_channels,semantic=True,positional=True,slice=True,uncertainty=True,rank=5):
#         super(CSAM,self).__init__()
#         self.semantic=semantic
#         self.positional=positional
#         self.slice=slice
#         if semantic:
#             self.semantic_att=SemanticAttentionModule(num_channels)
#         if positional:
#             self.positional_att=PositionalAttentionModule()
#         if slice:
#             self.slice_att=SliceAttentionModule(num_slices,uncertainty=uncertainty,rank=rank)
#     def forward(self,x):
#         if self.semantic:
#             x=self.semantic_att(x)
#         if self.positional:
#             x=self.positional_att(x)
#         if self.slice:
#             x=self.slice_att(x)
#         return x
#
#
#
# class CSAMConvBlock(nn.Module):
#     def __init__(self,input_channels,output_channels,max_pool,return_single=False):
#         super(CSAMConvBlock,self).__init__()
#         self.max_pool=max_pool
#         self.conv=[]
#         self.conv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
#         self.conv.append(nn.InstanceNorm2d(output_channels))
#         self.conv.append(nn.LeakyReLU())
#         self.conv.append(nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
#         self.conv.append(nn.InstanceNorm2d(output_channels))
#         self.conv.append(nn.LeakyReLU())
#         self.return_single=return_single
#         if max_pool:
#             self.pool=nn.MaxPool2d(2,stride=2,dilation=(1,1))
#         self.conv=nn.Sequential(*self.conv)
#
#     def forward(self,x):
#         x=self.conv(x)
#         b=x
#         if self.max_pool:
#             x=self.pool(x)
#         if self.return_single:
#             return x
#         else:
#             return x,b
#
#
# class CSAMDeconvBlock(nn.Module):
#     def __init__(self,input_channels,output_channels,intermediate_channels=-1):
#         super(CSAMDeconvBlock,self).__init__()
#         input_channels=int(input_channels)
#         output_channels=int(output_channels)
#         if intermediate_channels<0:
#             intermediate_channels=output_channels*2
#         else:
#             intermediate_channels=input_channels
#         self.upconv=[]
#         self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
#         self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=intermediate_channels//2,kernel_size=3,stride=1,padding=1))
#         self.conv=CSAMConvBlock(intermediate_channels,output_channels,False)
#         self.upconv=nn.Sequential(*self.upconv)
#
#     def forward(self,x,b):
#         x=self.upconv(x)
#         x=torch.cat((x,b),dim=1)
#         x,_=self.conv(x)
#         return x
#
# class UNetDecoder(nn.Module):
#     def __init__(self,num_layers,base_num):
#         super(UNetDecoder,self).__init__()
#         self.conv=[]
#         self.num_layers=num_layers
#         for i in range(num_layers-1,0,-1):
#             self.conv.append(CSAMDeconvBlock(base_num*(2**i),base_num*(2**(i-1))))
#         self.conv=nn.Sequential(*self.conv)
#     def forward(self,x,b):
#         for i in range(self.num_layers-1):
#             x=self.conv[i](x,b[i])
#         return x
#
# class EncoderCSAM(nn.Module):
#     def __init__(self,input_channels,num_layers,base_num,batch_size=20,semantic=True,positional=True,slice=True,uncertainty=True,rank=5):
#         super(EncoderCSAM,self).__init__()
#         self.conv=[]
#         self.num_layers=num_layers
#         for i in range(num_layers):
#             if i==0:
#                 self.conv.append(CSAMConvBlock(input_channels,base_num,True))
#             else:
#                 self.conv.append(CSAMConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1)))
#         self.conv=nn.Sequential(*self.conv)
#         self.attentions=[]
#         for i in range(num_layers):
#             self.attentions.append(CSAM(batch_size,base_num*(2**i),semantic,positional,slice,uncertainty,rank))
#         self.attentions=nn.Sequential(*self.attentions)
#
#     def forward(self,x):
#         b=[]
#         for i in range(self.num_layers):
#             x,block=self.conv[i](x)
#             if i!=self.num_layers-1:
#                 block=self.attentions[i](block)
#             else:
#                 x=self.attentions[i](x)
#             b.append(block)
#         b=b[:-1]
#         b=b[::-1]
#         return x,b
#
# class C2BAMUNet(nn.Module):
#     def __init__(self,input_channels,num_classes,num_layers,base_num=64,batch_size=20,semantic=True,positional=True,slice=True,uncertainty=False,rank=5):
#         super(C2BAMUNet,self).__init__()
#         self.encoder=EncoderCSAM(input_channels,num_layers,base_num,batch_size=batch_size,semantic=semantic,positional=positional,slice=slice,uncertainty=uncertainty,rank=rank)
#         self.decoder=UNetDecoder(num_layers,base_num)
#         self.base_num=base_num
#         self.input_channels=input_channels
#         self.num_classes=num_classes
#         self.conv_final=nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=(1,1))
#
#     def forward(self,x):
#         x,b=self.encoder(x)
#         x=self.decoder(x,b)
#         x=self.conv_final(x)
#         x=torch.softmax(x,dim=1)
#
#         return x
#
#
# class CSAMUNetPlusPlus(nn.Module):
#     def __init__(self,input_channels,num_classes,num_layers,base_num=64,batch_size=20,semantic=True,positional=True,slice=True,uncertainty=True,rank=5):
#         super(CSAMUNetPlusPlus).__init__()
#         self.num_layers=num_layers
#         nb_filter=[]
#         for i in range(num_layers):
#             nb_filter.append(base_num*(2**i))
#         self.pool=nn.MaxPool2d(2,2)
#         self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
#         self.conv=[]
#         for i in range(num_layers):
#             temp_conv=[]
#             for j in range(num_layers-i):
#                 if j==0:
#                     if i==0:
#                         inp=input_channels
#                     else:
#                         inp=nb_filter[i-1]
#                 else:
#                     inp=nb_filter[i]*j+nb_filter[i+1]
#                 temp_conv.append(CSAMConvBlock(inp,nb_filter[i],False,True))
#             self.conv.append(nn.Sequential(*temp_conv))
#         self.conv=nn.Sequential(*self.conv)
#         self.attentions=[]
#         for i in range(num_layers):
#             self.attentions.append(CSAM(batch_size,base_num*(2**i),semantic=semantic,positional=positional,slice=slice,uncertainty=uncertainty,rank=rank))
#         self.attentions=nn.Sequential(*self.attentions)
#         self.final=[]
#         for i in range(num_layers-1):
#             self.final.append(nn.Conv2d(nb_filter[0],num_classes,kernel_size=(1,1)))
#         self.final=nn.Sequential(*self.final)
#
#     def forward(self,inputs):
#         x=[]
#         for i in range(self.num_layers):
#             temp=[]
#             for j in range(self.num_layers-i):
#                 temp.append([])
#             x.append(temp)
#         x[0][0].append(self.conv[0][0](inputs))
#         for s in range(1,self.num_layers):
#             for i in range(s+1):
#                 if i==0:
#                     x[s-i][i].append(self.conv[s-i][i](self.pool(x[s-i-1][i][0])))
#                 else:
#                     for j in range(i):
#                         if j==0:
#                             block=x[s-i][j][0]
#                             block=self.attentions[s-i](block)
#                             temp_x=block
#                             #print(s-i,j)
#                         else:
#                             temp_x=torch.cat((temp_x,x[s-i][j][0]),dim=1)
#                             #print(s-i,j)
#                     temp_x=torch.cat((temp_x,self.up(x[s-i+1][i-1][0])),dim=1)
#                     #print('up',s-i+1,i-1,temp_x.size(),self.up(x[s-i+1][i-1][0]).size())
#                     x[s-i][i].append(self.conv[s-i][i](temp_x))
#         if self.training:
#             res=[]
#             for i in range(self.num_layers-1):
#                 res.append(self.final[i](x[0][i+1][0]))
#             return res
#         else:
#             return self.final[-1](x[0][-1][0])
# # #################################################################################################  CAT-Net  ######################################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import numpy as np
#
# class CrossSliceAttention(nn.Module):
#     def __init__(self,input_channels):
#         super(CrossSliceAttention,self).__init__()
#         self.linear_q=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
#         self.linear_k=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
#         self.linear_v=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
#
#     def forward(self,pooled_features,features):
#         q=self.linear_q(pooled_features)
#         q=q.view(q.size(0),-1)
#         k=self.linear_k(pooled_features)
#         k=k.view(k.size(0),-1)
#         v=self.linear_v(features)
#         x=torch.matmul(q,k.permute(1,0))/np.sqrt(q.size(1))
#         x=torch.softmax(x,dim=1)
#         out=torch.zeros_like(v)
#         for i in range(x.size(0)):
#             temp=x[i,:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#             out[i,:,:,:]=torch.sum(temp*v,dim=0).clone()
#         return out
#
#
# class MultiHeadedCrossSliceAttentionModule(nn.Module):
#     def __init__(self,input_channels,heads=3,pool_kernel_size=(4,4),input_size=(128,512),batch_size=4,pool_method='avgpool'):
#         super(MultiHeadedCrossSliceAttentionModule,self).__init__()
#         self.attentions=[]
#         self.linear1=nn.Conv2d(in_channels=heads*input_channels,out_channels=input_channels,kernel_size=(1,1))
#         self.norm1=nn.LayerNorm([batch_size,input_channels,input_size[0],input_size[1]])
#         self.linear2=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1))
#         self.norm2=nn.LayerNorm([batch_size,input_channels,input_size[0],input_size[1]])
#
#         if pool_method=="maxpool":
#             self.pool=nn.MaxPool2d(kernel_size=pool_kernel_size)
#         elif pool_method=="avgpool":
#             self.pool=nn.AvgPool2d(kernel_size=pool_kernel_size)
#         else:
#             assert (False)  # not implemented yet
#
#         for i in range(heads):
#             self.attentions.append(CrossSliceAttention(input_channels))
#         self.attentions=nn.Sequential(*self.attentions)
#
#     def forward(self,pooled_features,features):
#
#         for i in range(len(self.attentions)):
#             x_=self.attentions[i](pooled_features,features)
#             if i==0:
#                 x=x_
#             else:
#                 x=torch.cat((x,x_),dim=1)
#         out=self.linear1(x)
#         x=F.gelu(out)+features
#         out_=self.norm1(x)
#         out=self.linear2(out_)
#         x=F.gelu(out)+out_
#         out=self.norm2(x)
#         pooled_out=self.pool(out)
#         return pooled_out,out
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self,d_model,is_pe_learnable=True,max_len=20):
#         super(PositionalEncoding,self).__init__()
#
#         position=torch.arange(max_len).unsqueeze(1)
#         div_term=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
#         pe=torch.zeros(max_len,d_model,1,1)
#         pe[:,0::2,0,0]=torch.sin(position*div_term)
#         pe[:,1::2,0,0]=torch.cos(position*div_term)
#         self.pe=nn.Parameter(pe.clone(),is_pe_learnable)
#         #self.register_buffer('pe',self.pe)
#
#     def forward(self,x):
#         return x+self.pe[:x.size(0),:,:,:]
#
#     def get_pe(self):
#         return self.pe[:,:,0,0]
#
# class ConvBlock(nn.Module):
#     def __init__(self,input_channels,output_channels,max_pool,return_single=False):
#         super(ConvBlock,self).__init__()
#         self.max_pool=max_pool
#         self.conv=[]
#         self.conv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
#         self.conv.append(nn.InstanceNorm2d(output_channels))
#         self.conv.append(nn.LeakyReLU())
#         self.conv.append(nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
#         self.conv.append(nn.InstanceNorm2d(output_channels))
#         self.conv.append(nn.LeakyReLU())
#         self.return_single=return_single
#         if max_pool:
#             self.pool=nn.MaxPool2d(2,stride=2,dilation=(1,1))
#         self.conv=nn.Sequential(*self.conv)
#
#     def forward(self,x):
#         x=self.conv(x)
#         b=x
#         if self.max_pool:
#             x=self.pool(x)
#         if self.return_single:
#             return x
#         else:
#             return x,b
#
#
# class DeconvBlock(nn.Module):
#     def __init__(self,input_channels,output_channels,intermediate_channels=-1):
#         super(DeconvBlock,self).__init__()
#         input_channels=int(input_channels)
#         output_channels=int(output_channels)
#         if intermediate_channels<0:
#             intermediate_channels=output_channels*2
#         else:
#             intermediate_channels=input_channels
#         self.upconv=[]
#         self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
#         self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=intermediate_channels//2,kernel_size=3,stride=1,padding=1))
#         self.conv=ConvBlock(intermediate_channels,output_channels,False)
#         self.upconv=nn.Sequential(*self.upconv)
#
#     def forward(self,x,b):
#         x=self.upconv(x)
#         x=torch.cat((x,b),dim=1)
#         x,_=self.conv(x)
#         return x
#
# class UNetDecoder(nn.Module):
#     def __init__(self,num_layers,base_num):
#         super(UNetDecoder,self).__init__()
#         self.conv=[]
#         self.num_layers=num_layers
#         for i in range(num_layers-1,0,-1):
#             self.conv.append(DeconvBlock(base_num*(2**i),base_num*(2**(i-1))))
#         self.conv=nn.Sequential(*self.conv)
#
#     def forward(self,x,b):
#         for i in range(self.num_layers-1):
#             x=self.conv[i](x,b[i])
#         return x
#
# class CrossSliceUNetEncoder(nn.Module):
#     def __init__(self,input_channels,num_layers,base_num,num_attention_blocks=3,heads=4,pool_kernel_size=(4,4),input_size=(128,512),batch_size=4,pool_method='avgpool',is_pe_learnable=True):
#         super(CrossSliceUNetEncoder,self).__init__()
#         self.conv=[]
#         self.num_layers=num_layers
#         self.num_attention_blocks=num_attention_blocks
#         for i in range(num_layers):
#             if i==0:
#                 self.conv.append(ConvBlock(input_channels,base_num,True))
#             else:
#                 self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1)))
#         self.conv=nn.Sequential(*self.conv)
#         self.pools=[]
#         self.pes=[]
#         self.attentions=[]
#         for i in range(num_layers):
#             if pool_method=='maxpool':
#                 self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
#             elif pool_method=='avgpool':
#                 self.pools.append(nn.AvgPool2d(kernel_size=pool_kernel_size))
#             else:
#                 assert (False)  # not implemented yet
#
#             self.pes.append(PositionalEncoding(base_num*(2**i),is_pe_learnable))
#             temp=[]
#             for j in range(num_attention_blocks):
#                 temp.append(MultiHeadedCrossSliceAttentionModule(base_num*(2**i),heads,pool_kernel_size,input_size,batch_size,pool_method))
#             input_size=(input_size[0]//2,input_size[1]//2)
#             self.attentions.append(nn.Sequential(*temp))
#         self.attentions=nn.Sequential(*self.attentions)
#         self.pes=nn.Sequential(*self.pes)
#
#     def forward(self,x):
#         b=[]
#         for i in range(self.num_layers):
#             x,block=self.conv[i](x)
#             if i!=self.num_layers-1:
#                 block=self.pes[i](block)
#                 block_pool=self.pools[i](block)
#                 for j in range(self.num_attention_blocks):
#                     block_pool,block=self.attentions[i][j](block_pool,block)
#             else:
#                 x=self.pes[i](x)
#                 x_pool=self.pools[i](x)
#                 for j in range(self.num_attention_blocks):
#                     x_pool,x=self.attentions[i][j](x_pool,x)
#             b.append(block)
#         b=b[:-1]
#         b=b[::-1]
#         return x,b
#
# class CrossSliceAttentionUNet(nn.Module):
#     # AxR 256 512 LCR 128 512
#     def __init__(self,input_channels,num_classes,num_layers,heads=3,num_attention_blocks=2,base_num=64,pool_kernel_size=(4,4),input_size=(128,512),batch_size=4,pool_method="avgpool",is_pe_learnable=True):
#         super(CrossSliceAttentionUNet,self).__init__()
#         self.encoder=CrossSliceUNetEncoder(input_channels,num_layers,base_num,num_attention_blocks,heads,pool_kernel_size,input_size,batch_size,pool_method,is_pe_learnable)
#         self.decoder=UNetDecoder(num_layers,base_num)
#         self.base_num=base_num
#         self.input_channels=input_channels
#         self.num_classes=num_classes
#         self.conv_final=nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=(1,1))
#
#     def forward(self,x):
#         x,b=self.encoder(x)
#         x=self.decoder(x,b)
#         x=self.conv_final(x)
#         x = torch.softmax(x, dim=1)
#         return x
#
#
# class CrossSliceUNetPlusPlus(nn.Module):
#     def __init__(self,input_channels,num_classes,num_layers,heads=3,num_attention_blocks=2,base_num=64,pool_kernel_size=(4,4),input_size=(128,512),batch_size=4,pool_method="maxpool",is_pe_learnable=True):
#         super(CrossSliceUNetPlusPlus).__init__()
#         self.num_layers=num_layers
#         self.num_attention_blocks=num_attention_blocks
#         nb_filter=[]
#         for i in range(num_layers):
#             nb_filter.append(base_num*(2**i))
#         self.pool=nn.MaxPool2d(2,2)
#         self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
#         self.conv=[]
#         for i in range(num_layers):
#             temp_conv=[]
#             for j in range(num_layers-i):
#                 if j==0:
#                     if i==0:
#                         inp=input_channels
#                     else:
#                         inp=nb_filter[i-1]
#                 else:
#                     inp=nb_filter[i]*j+nb_filter[i+1]
#                 temp_conv.append(ConvBlock(inp,nb_filter[i],False,True))
#             self.conv.append(nn.Sequential(*temp_conv))
#         self.conv=nn.Sequential(*self.conv)
#         self.pools=[]
#         self.pes=[]
#         self.attentions=[]
#         for i in range(num_layers):
#             if pool_method=='maxpool':
#                 self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
#             elif pool_method=='avgpool':
#                 self.pools.append(nn.AvgPool2d(kernel_size=pool_kernel_size))
#             else:
#                 assert (False)  # not implemented yet
#
#             self.pes.append(PositionalEncoding(base_num*(2**i),is_pe_learnable))
#             temp=[]
#             for j in range(num_attention_blocks):
#                 temp.append(MultiHeadedCrossSliceAttentionModule(base_num*(2**i),heads,pool_kernel_size,input_size,batch_size,pool_method))
#             input_size=(input_size[0]//2,input_size[1]//2)
#             self.attentions.append(nn.Sequential(*temp))
#         self.attentions=nn.Sequential(*self.attentions)
#         self.pes=nn.Sequential(*self.pes)
#         self.final=[]
#         for i in range(num_layers-1):
#             self.final.append(nn.Conv2d(nb_filter[0],num_classes,kernel_size=(1,1)))
#         self.final=nn.Sequential(*self.final)
#
#     def forward(self,inputs):
#         x=[]
#         for i in range(self.num_layers):
#             temp=[]
#             for j in range(self.num_layers-i):
#                 temp.append([])
#             x.append(temp)
#         x[0][0].append(self.conv[0][0](inputs))
#         for s in range(1,self.num_layers):
#             for i in range(s+1):
#                 if i==0:
#                     x[s-i][i].append(self.conv[s-i][i](self.pool(x[s-i-1][i][0])))
#                 else:
#                     for j in range(i):
#                         if j==0:
#                             block=x[s-i][j][0]
#                             block_pool=self.pools[s-i](block)
#                             for k in range(self.num_attention_blocks):
#                                 block_pool,block=self.attentions[s-i][k](block_pool,block)
#                             temp_x=block
#                             #print(s-i,j)
#                         else:
#                             temp_x=torch.cat((temp_x,x[s-i][j][0]),dim=1)
#                             #print(s-i,j)
#                     temp_x=torch.cat((temp_x,self.up(x[s-i+1][i-1][0])),dim=1)
#                     #print('up',s-i+1,i-1,temp_x.size(),self.up(x[s-i+1][i-1][0]).size())
#                     x[s-i][i].append(self.conv[s-i][i](temp_x))
#         if self.training:
#             res=[]
#             for i in range(self.num_layers-1):
#                 res.append(self.final[i](x[0][i+1][0]))
#             return res
#         else:
#             return self.final[-1](x[0][-1][0])