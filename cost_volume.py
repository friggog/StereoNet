import torch
import numpy as np


def CostVolume(input_feature, candidate_feature, position="left", k=3, batch_size=1, channel=32, D=192, H=256, W=512):
    """
    Some parameters:
        position
            means whether the input feature img is left or right
        k
            the conv counts of the first stage, the feature extraction stage
    """
    origin = input_feature  # img shape : [batch_size, H // 2**k, W // 2**k, channel]
    candidate = candidate_feature
    """ if the input image is the left image, and needs to compare with the right candidate.
        Then it should move to right and pad in left"""
    if position == "left":
        leftMinusRightMove_List = []
        for disparity in range(D // 2**k):
            if disparity == 0:
                leftMinusRightMove = origin - candidate
                leftMinusRightMove_List.append(leftMinusRightMove)
            else:
                zero_padding = np.zeros((batch_size, H // 2**k, disparity, channel))

                right_move = np.concatenate([zero_padding, origin], axis=2)
                leftMinusRightMove = right_move[:, :, W // 2**k, :] - candidate
                leftMinusRightMove_List.append(leftMinusRightMove)
        cost_volume = np.stack(leftMinusRightMove_List, axis=1)
        return cost_volume

    elif position == "right":
        rightMinusLeftMove_List = []
        for disparity in range(D // 2**k):
            if disparity == 0:
                rightMinusLeftMove = origin - candidate
                rightMinusLeftMove_List.append(rightMinusLeftMove)
            else:
                zero_padding = np.zeros((batch_size, H // 2**k, disparity, channel))

                left_move = np.concatenate([zero_padding, origin], axis=2)
                rightMinusLeftMove = left_move[:, :, W // 2**k, :] - candidate
                rightMinusLeftMove_List.append(rightMinusLeftMove)
        cost_volume = np.stack(rightMinusLeftMove_List, axis=1)
        return cost_volume



