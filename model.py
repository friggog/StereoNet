import torch.nn as nn
import torch
import torch.nn.functional as F
from cost_volume import CostVolume
import numpy as np


class StereoNet(nn.Module):
    def __init__(self):
        super(StereoNet, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.Conv2d(32, 32, 5, stride=2),
        )

        self.res = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            nn.Conv2d(32, 32, 3),
        )

        """ using 3d conv to instead the Euclidean distance"""
        self.cost_volume_filter = nn.Sequential(
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            nn.Conv3d(32, 1, 3),
        )

        # TODO what is the input channel size
        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, 3),
            ResBlock(32, 32, padding=1, dilation=1),
            ResBlock(32, 32, padding=2, dilation=2),
            ResBlock(32, 32, padding=4, dilation=4),
            ResBlock(32, 32, padding=8, dilation=8),
            ResBlock(32, 32, padding=1, dilation=1),
            ResBlock(32, 32, padding=1, dilation=1),
            nn.Conv2d(32, 1, 3),
        )

    def forward_once_1(self, x):
        output = self.downsampling(x)
        output = self.res(output)

        return output

    def forward_stage1(self, input_l, input_r):
        output_l = self.forward_once_1(input_l)
        output_r = self.forward_once_1(input_r)

        return output_l, output_r

    def forward_once_2(self, cost_volume):
        output = self.cost_volume_filter(cost_volume)
        disparity_low = soft_argmin(output)

        return disparity_low  # low resolution disparity map

    def forward_stage2(self, feature_l, feature_r):
        cost_v_l = CostVolume(feature_l, feature_r, "left", k=4)
        cost_v_r = CostVolume(feature_r, feature_l, "right", k=4)
        disparity_low_l = self.forward_once_2(cost_v_l)
        disparity_low_r = self.forward_once_2(cost_v_r)

        return disparity_low_l, disparity_low_r

    def forward_once3(self, concatenation):
        refined_d = self.refine(concatenation)

        return refined_d

    def forward_stage3(self, disparity_low_l, disparity_low_r, left, right):
        """upsample and concatenate"""
        d_high_l = nn.functional.upsample_bilinear(disparity_low_l, left.shape)
        d_high_r = nn.functional.upsample_bilinear(disparity_low_r, right.shape)

        d_concat_l = np.concatenate([d_high_l, left], axis=3)
        d_concat_r = np.concatenate([d_high_r, right], axis=3)

        d_refined_l = self.forward_once3(d_concat_l)
        d_refined_r = self.forward_once3(d_concat_r)

        return d_refined_l, d_refined_r

    def forward(self, left, right):
        left_feature, right_feature = self.forward_stage1(left, right)
        disparity_low_l, disparity_low_r = self.forward_stage2(left_feature, right_feature)
        d_refined_l, d_refined_r = self.forward_stage3(disparity_low_l, disparity_low_r, left, right)

        d_final_l = nn.ReLU(disparity_low_r + d_refined_l)
        d_final_r = nn.ReLU(disparity_low_r + d_refined_r)

        return d_final_l, d_final_r


class MetricBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(MetricBlock, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channel, 3)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1, dilation=0, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def soft_argmin(cost_volume, D=192):

    """Remove single-dimensional entries from the shape of an array."""
    cost_volume_D_squeeze = np.squeeze(cost_volume, axis=[0, 4])  # 192 256 512
    softmax = nn.Softmax(dim=0)
    disparity_softmax = softmax(cost_volume_D_squeeze)   # 192 256 512
    # disparity_softmax = nn.Softmax(cost_volume_D_squeeze, dim=0)

    d_grid = np.cast(np.arange(D), np.float32)
    d_grid = np.reshape(d_grid, (-1, 1, 1))
    d_grid = np.tile(d_grid, [1, 256, 512])

    arg_soft_min = np.sum(disparity_softmax*d_grid, axis=0, keepdims=True)

    return arg_soft_min



