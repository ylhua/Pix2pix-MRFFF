# import os
# import time
# import cv2
# import numpy as np
# import tensorflow as tf
# import scipy.misc
# import glob
# import pynvml
#
# from skimage.measure import compare_psnr
# from skimage.measure import compare_ssim
#
# total_ssim = 0
# total_psnr = 0
# psnr_weight = 1 / 20
# ssim_weight = 1
# val_A = '../results/enhance_pix2pix/test_latest/fake_B/'
# val_B = '../results/enhance_pix2pix/test_latest/real_B/'
# dataA = glob.glob(val_A + '*.png')
#
#
# for i in range(len(dataA)):
#     image_name = dataA[i].split('/')[-1]
#     image_name_B = image_name.replace('fake', 'real')
#     img_A = cv2.imread(os.path.join(val_A, image_name))
#     img_A = np.expand_dims(img_A, axis=0)
#     # x = np.expand_dims(x, axis=0)/255*2 - 1
#
#     img_B = cv2.imread(os.path.join(val_B, image_name_B))
#     img_B = np.expand_dims(img_B, axis=0)
#     # y = y / 255 * 2 - 1
#     #
#     # generated_B = (((generated_B[0] + 1) / 2) * 255).astype(np.uint8)
#     # real_B = (((y + 1) / 2) * 255).astype(np.uint8)
#     # real_B = img_A.astype(np.uint8)
#     # generated_B = img_B.astype(np.uint8)
#     real_B = img_A
#     generated_B = img_B
#     psnr = compare_psnr(real_B, generated_B)
#     ssim = compare_ssim(real_B, generated_B, multichannel=True)
#
#     total_psnr = total_psnr + psnr
#     total_ssim = total_ssim + ssim
#
# average_psnr = total_psnr / len(dataA)
# average_ssim = total_ssim / len(dataA)
#
# score = average_psnr * psnr_weight + average_ssim * ssim_weight

import os
import cv2

import numpy as np


from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

GT_dir='/media/linger/udata/pycharmprojects/pytorch-CycleGAN-and-pix2pix-master/results/unet4/test_latest/real/'
input_dir='/media/linger/udata/pycharmprojects/pytorch-CycleGAN-and-pix2pix-master/results/unet5_unet++/test_latest/fake/'

total_ssim = 0
total_psnr = 0
psnr_weight = 1/20
ssim_weight = 1

GT_list = os.listdir(GT_dir)
input_list = os.listdir(input_dir)

for i, img_file in enumerate(input_list, 1):
    img = cv2.imread(os.path.join(input_dir, img_file), 1)
    GT_file = img_file.replace('fake', 'real')
    GT = cv2.imread(os.path.join(GT_dir, GT_file), 1).astype(np.uint8)

    psnr = compare_psnr(GT, img)
    ssim = compare_ssim(GT, img, multichannel = True)

    total_psnr = total_psnr + psnr
    total_ssim = total_ssim + ssim

average_psnr = total_psnr / len(GT_list)
average_ssim = total_ssim / len(GT_list)

score = average_psnr * psnr_weight + average_ssim * ssim_weight

line = 'Score: %.6f, PSNR: %.6f, SSIM: %.6f' %(score, average_psnr, average_ssim)
print(line)







# import torch
# import torch.nn as nn
# from torch.nn import init
# import functools
# from torch.optim import lr_scheduler
# import torch.nn.functional as F
# from torch import nn
# from collections import OrderedDict
# from torch.nn import BatchNorm2d as bn
#
#
# class DenseASPP(nn.Module):
#     """
#     * output_scale can only set as 8 or 16
#     """
#     def __init__(self, model_cfg, n_class=19, output_stride=8):
#         super(DenseASPP, self).__init__()
#
#         num_init_features = 64
#
#         dropout0 = 0.1
#         dropout1 = 0.1
#         d_feature0 = 128
#         d_feature1 = 64
#
#         # Each denseblock
#         num_features = num_init_features
#
#         self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
#                                       dilation_rate=3, drop_out=dropout0, bn_start=False)
#
#         self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
#                                       dilation_rate=6, drop_out=dropout0, bn_start=True)
#
#         self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
#                                        dilation_rate=12, drop_out=dropout0, bn_start=True)
#
#         self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
#                                        dilation_rate=18, drop_out=dropout0, bn_start=True)
#
#         self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
#                                        dilation_rate=24, drop_out=dropout0, bn_start=True)
#         num_features = num_features + 5 * d_feature1
#
#         self.classification = nn.Sequential(
#             nn.Dropout2d(p=dropout1),
#             nn.Conv2d(in_channels=num_features, out_channels=n_class, kernel_size=1, padding=0),
#             nn.Upsample(scale_factor=8, mode='bilinear'),
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform(m.weight.data)
#
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, _input):
#         feature = self.features(_input)
#
#         aspp3 = self.ASPP_3(feature)
#         feature = torch.cat((aspp3, feature), dim=1)
#
#         aspp6 = self.ASPP_6(feature)
#         feature = torch.cat((aspp6, feature), dim=1)
#
#         aspp12 = self.ASPP_12(feature)
#         feature = torch.cat((aspp12, feature), dim=1)
#
#         aspp18 = self.ASPP_18(feature)
#         feature = torch.cat((aspp18, feature), dim=1)
#
#         aspp24 = self.ASPP_24(feature)
#         feature = torch.cat((aspp24, feature), dim=1)
#
#         cls = self.classification(feature)
#
#         return cls
#
#
# class _DenseAsppBlock(nn.Sequential):
#     """ ConvNet block for building DenseASPP. """
#
#     def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
#         super(_DenseAsppBlock, self).__init__()
#         if bn_start:
#             self.add_module('norm.1', bn(input_num, momentum=0.0003)),
#
#         self.add_module('relu.1', nn.ReLU(inplace=True)),
#         self.add_module('conv.1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),
#
#         self.add_module('norm.2', bn(num1, momentum=0.0003)),
#         self.add_module('relu.2', nn.ReLU(inplace=True)),
#         self.add_module('conv.2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
#                                             dilation=dilation_rate, padding=dilation_rate)),
#
#         self.drop_rate = drop_out
#
#     def forward(self, _input):
#         feature = super(_DenseAsppBlock, self).forward(_input)
#
#         if self.drop_rate > 0:
#             feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
#
#         return feature
#
#
# class _DenseLayer(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation_rate=1):
#         super(_DenseLayer, self).__init__()
#         self.add_module('norm.1', bn(num_input_features)),
#         self.add_module('relu.1', nn.ReLU(inplace=True)),
#         self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
#                         growth_rate, kernel_size=1, stride=1, bias=False)),
#         self.add_module('norm.2', bn(bn_size * growth_rate)),
#         self.add_module('relu.2', nn.ReLU(inplace=True)),
#         self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
#                         kernel_size=3, stride=1, dilation=dilation_rate, padding=dilation_rate, bias=False)),
#         self.drop_rate = drop_rate
#
#     def forward(self, x):
#         new_features = super(_DenseLayer, self).forward(x)
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
#         return torch.cat([x, new_features], 1)
#
#
# class _DenseBlock(nn.Sequential):
#     def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation_rate=1):
#         super(_DenseBlock, self).__init__()
#         for i in range(num_layers):
#             layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
#                                 bn_size, drop_rate, dilation_rate=dilation_rate)
#             self.add_module('denselayer%d' % (i + 1), layer)
#
#
# class _Transition(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features, stride=2):
#         super(_Transition, self).__init__()
#         self.add_module('norm', bn(num_input_features))
#         self.add_module('relu', nn.ReLU(inplace=True))
#         self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
#         if stride == 2:
#             self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))