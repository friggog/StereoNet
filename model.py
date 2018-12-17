import torch.nn as nn
import torch.nn.functional as F
from cost_volume import CostVolume

class SteroNet(nn.Module):
    def __init__(self):
        super(SteroNet, self).__init__()
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

        self.refine = nn.Sequential(

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
        disparity_low = argmin(output)

        return disparity_low  # low resolution disparity map

    def forward_stage2(self, feature_l, feature_r):
        cost_v_l = CostVolume(feature_l, feature_r, "left", k=4)
        cost_v_r = CostVolume(feature_r, feature_l, "right", k=4)
        disparity_low_l = self.forward_once_2(cost_v_l)
        disparity_low_r = self.forward_once_2(cost_v_r)

        return disparity_low_l, disparity_low_r


    def forward(self, left, right):
        left_feature, right_feature = self.forward_stage1(left, right)
        disparity_low_l, disparity_low_r = self.forward_stage2(left_feature, right_feature)


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
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)
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
