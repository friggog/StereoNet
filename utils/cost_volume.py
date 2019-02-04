import torch
import numpy as np


def CostVolume(input_feature, candidate_feature, position="left", method="subtract", k=4, channel=32, D=256):
    origin = input_feature
    candidate = candidate_feature
    """ if the input image is the left image, and needs to compare with the right candidate.
        Then it should move to right and pad in left"""
    if position == "left":
        leftMinusRightMove_List = []
        for disparity in range(D // 2**k):
            if disparity == 0:
                if method == "subtract":
                    leftMinusRightMove = origin - candidate
                else:
                    leftMinusRightMove = torch.cat((origin, candidate), 1)
                leftMinusRightMove_List.append(leftMinusRightMove)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding.cuda()

                right_move = torch.cat((zero_padding, origin), 3)

                if method == "subtract":
                    leftMinusRightMove = right_move[:, :, :, :origin.shape[3]] - candidate
                else:
                    leftMinusRightMove = torch.cat((right_move[:, :, :, :origin.shape[3]], candidate), 1)

                leftMinusRightMove_List.append(leftMinusRightMove)
        cost_volume = torch.stack(leftMinusRightMove_List, dim=1)
        return cost_volume
    elif position == "right":
        rightMinusLeftMove_List = []
        for disparity in range(D // 2**k):
            if disparity == 0:
                if method == "subtract":
                    rightMinusLeftMove = origin - candidate
                else:
                    rightMinusLeftMove = torch.cat((origin, candidate), 1)
                rightMinusLeftMove_List.append(rightMinusLeftMove)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding.cuda()

                left_move = torch.cat((origin, zero_padding), 3)

                if method == "subtract":
                    rightMinusLeftMove = left_move[:, :, :, -origin.shape[3]:] - candidate
                else:
                    rightMinusLeftMove = torch.cat((left_move[:, :, :, -origin.shape[3]:], candidate), 1)

                rightMinusLeftMove_List.append(rightMinusLeftMove)
        cost_volume = torch.stack(rightMinusLeftMove_List, dim=1)
        return cost_volume
    else:
        raise Exception('invalid cost volume direction')
