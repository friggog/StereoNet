import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.cost_volume import CostVolume
import torchvision.transforms as transforms


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
            raise Exception("cost_volume_method is not right")

        self.downsampling = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            # nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
        )

        self.res = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            # ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
        )

        """ using 3d conv to instead the Euclidean distance"""
        self.cost_volume_filter = nn.Sequential(
            MetricBlock(cost_volume_channel, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            nn.Conv3d(32, 1, 3, padding=1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=2),
            ResBlock(32, 32, dilation=4),
            ResBlock(32, 32, dilation=8),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=1),
            nn.Conv2d(32, 1, 3, padding=1)
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
        """the index cost volume's dimension is not right for conv3d here, so we change it"""
        cost_volume = cost_volume.permute([0, 2, 1, 3, 4])
        output = self.cost_volume_filter(cost_volume)
        disparity_low = soft_argmin(output)
        return disparity_low  # low resolution disparity map

    def forward_stage2(self, feature_l, feature_r):
        cost_v_l = CostVolume(feature_l, feature_r, position='left', method=self.cost_volume_method, k=3)
        cost_v_r = CostVolume(feature_r, feature_l, position='right', method=self.cost_volume_method, k=3)
        disparity_low_l = self.forward_once_2(cost_v_l)
        disparity_low_r = self.forward_once_2(cost_v_r)
        return disparity_low_l, disparity_low_r

    def forward_stage3(self, disparity_low_l, disparity_low_r, left, right):
        """upsample and concatenate"""
        d_high_l = nn.functional.interpolate(disparity_low_l, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        d_high_r = nn.functional.interpolate(disparity_low_r, [right.shape[2], right.shape[3]], mode='bilinear', align_corners=True)

        d_concat_l = torch.cat([d_high_l, left], dim=1)
        d_concat_r = torch.cat([d_high_r, right], dim=1)

        d_refined_l = self.refine(d_concat_l)
        d_refined_r = self.refine(d_concat_r)

        return d_refined_l, d_refined_r

    def forward(self, left, right):
        left_feature, right_feature = self.forward_stage1(left, right)
        disparity_low_l, disparity_low_r = self.forward_stage2(
            left_feature, right_feature)
        d_refined_l, d_refined_r = self.forward_stage3(
            disparity_low_l, disparity_low_r, left, right)

        # tensor = d_refined_l
        # t = transforms.ToPILImage()
        # image = disparity_low_l[0].cpu().clone()
        # image = image.squeeze(1)
        # image = t(image)
        # image.show()
        # input()

        d_high_l = nn.functional.interpolate(disparity_low_l, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        d_high_r = nn.functional.interpolate(disparity_low_r, [right.shape[2], right.shape[3]], mode='bilinear', align_corners=True)

        d_final_l = nn.ReLU()(d_high_l + d_refined_l)
        d_final_r = nn.ReLU()(d_high_r + d_refined_r)

        return d_high_l, d_high_r, d_final_l, d_final_r


class MetricBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
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
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out


def soft_argmin(cost_volume):
    """Remove single-dimensional entries from the shape of an array."""
    cost_volume_D_squeeze = torch.squeeze(cost_volume, dim=1)

    softmax = nn.Softmax(dim=1)
    disparity_softmax = softmax(cost_volume_D_squeeze)

    d_grid = torch.arange(cost_volume_D_squeeze.shape[1], dtype=torch.float)
    d_grid = d_grid.reshape(-1, 1, 1)
    d_grid = d_grid.repeat((cost_volume.shape[0], 1, cost_volume.shape[3], cost_volume.shape[4]))
    d_grid = d_grid.to('cuda')

    tmp = disparity_softmax*d_grid
    arg_soft_min = torch.sum(tmp, dim=1, keepdim=True)

    return arg_soft_min
