import torch.nn as nn
import torch
import torch.nn.functional as F
from cost_volume import CostVolume
import numpy as np
from PIL import Image
from torchvision import datasets, transforms


class StereoNet(nn.Module):
    def __init__(self, batch_size, cost_volume_method):
        super(StereoNet, self).__init__()

        self.batch_size = batch_size
        self.cost_volume_method = cost_volume_method
        cost_volume_channel = 32
        if cost_volume_method == "subtract":
            cost_volume_channel = 32
        elif cost_volume_method == "concat":
            cost_volume_channel = 64
        else:
            print("cost_volume_method is not right")

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
            MetricBlock(cost_volume_channel, 32),
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

        # print(x.shape) # torch.Size([4, 3, 375, 1242])
        # print(output.shape) # torch.Size([4, 32, 18, 72])

        return output

    def forward_stage1(self, input_l, input_r):
        output_l = self.forward_once_1(input_l)
        output_r = self.forward_once_1(input_r)

        return output_l, output_r

    def forward_once_2(self, cost_volume):
        """the index cost volume's dimension is not right for conv3d here, so we change it"""
        cost_volume = cost_volume.permute([0, 2, 1, 3, 4])
        # print("cost_volume.shape")
        # print(cost_volume.shape)
        output = self.cost_volume_filter(cost_volume)
        # print(output.shape)
        disparity_low = soft_argmin(output, self.batch_size)

        return disparity_low  # low resolution disparity map

    def forward_stage2(self, feature_l, feature_r):
        cost_v_l = CostVolume(feature_l, feature_r, "left", method=self.cost_volume_method, k=4, batch_size=self.batch_size)
        # cost_v_r = CostVolume(feature_r, feature_l, "right", k=4, batch_size=self.batch_size)
        disparity_low = self.forward_once_2(cost_v_l)
        # disparity_low_r = self.forward_once_2(cost_v_r)

        return disparity_low

    def forward_once3(self, concatenation):
        refined_d = self.refine(concatenation)

        return refined_d

    def forward_stage3(self, disparity_low, left):
        """upsample and concatenate"""
        # print(left.shape)

        # d_high_l = nn.functional.upsample_bilinear(disparity_low_l, [left.shape[2], left.shape[3]])
        d_high = nn.functional.interpolate(disparity_low, [left.shape[2], left.shape[3]])
        # d_high_r = nn.functional.upsample_bilinear(disparity_low_r, [right.shape[2], right.shape[3]])
        # d_high_r = nn.functional.interpolate(disparity_low_r, [right.shape[2], right.shape[3]])

        # print(disparity_low_l.shape) # torch.Size([4, 1, 16, 70])
        # print(d_high_l.shape) # torch.Size([4, 1, 375, 1242])
        # print(left.shape)

        d_concat = torch.cat([d_high, left], dim=1)
        # d_concat_r = torch.cat([d_high_r, right], dim=1)

        d_refined = self.forward_once3(d_concat)
        # d_refined_r = self.forward_once3(d_concat_r)

        return d_refined

    def forward(self, left, right):
        left_feature, right_feature = self.forward_stage1(left, right)
        disparity_low = self.forward_stage2(left_feature, right_feature)
        d_refined = self.forward_stage3(disparity_low, left)

        # print(d_refined_l.shape)

        # tensor = d_refined_l
        # unloader = transforms.ToPILImage()
        # image = tensor.cpu().clone()
        # image = image.squeeze(1)
        # image = unloader(image)
        # image.show()

        # TODO : the paper says here should add them up, but I don't agree with that
        # d_final_l = nn.ReLU(disparity_low_r + d_refined_l)
        # d_final_r = nn.ReLU(disparity_low_r + d_refined_r)

        # return d_final_l, d_final_r
        return d_refined

class MetricBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(MetricBlock, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1, dilation=1, stride=1, downsample=None):
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

        # print(out.shape) # torch.Size([4, 32, 20, 74])
        # print(residual.shape) # torch.Size([4, 32, 20, 74])
        out += residual
        out = self.relu(out)

        return out


def soft_argmin(cost_volume, batch_size, D=192):

    """Remove single-dimensional entries from the shape of an array."""
    # print(cost_volume.shape)
    cost_volume_D_squeeze = torch.squeeze(cost_volume)  # 10 16 70
    softmax = nn.Softmax(dim=0)
    disparity_softmax = softmax(cost_volume_D_squeeze)   # 10 16 70
    # disparity_softmax = nn.Softmax(cost_volume_D_squeeze, dim=0)

    # d_grid = np.cast(np.arange(D), np.float32)
    # d_grid = np.reshape(d_grid, (-1, 1, 1))
    d_grid = torch.arange(cost_volume_D_squeeze.shape[1], dtype=torch.float)
    # print(d_grid.shape)
    d_grid = d_grid.reshape(-1, 1, 1)
    # d_grid = np.tile(d_grid, [1, 256, 512])
    d_grid = d_grid.repeat((batch_size, 1, 16, 70))
    d_grid = d_grid.to('cuda')

    # print(disparity_softmax.shape)
    # print(d_grid.shape) # torch.Size([4, 10, 16, 70])  batch_size=4, disparity=10, H=16, W=70 ???
    tmp = disparity_softmax*d_grid
    arg_soft_min = torch.sum(tmp, dim=1, keepdim=True)  # TODO originally in GC-NET the dim=0 here (tf version)

    return arg_soft_min



