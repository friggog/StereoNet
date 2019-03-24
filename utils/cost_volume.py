import torch
import numpy as np

import torchvision.transforms as transforms

def CostVolume(input_feature, candidate_feature, position="left", method="subtract", k=4, channel=32, D=256):
    if position != "left" and position != "right":
        raise Exception('invalid cost volume direction')
    origin = input_feature
    candidate = candidate_feature
    oMinusM_List = []
    if position == "left":
        for disparity in range(D // 2**k):
            if disparity == 0:
                if method == "subtract":
                    oMinusM = origin - candidate
                else:
                    oMinusM = torch.cat((origin, candidate), 1)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding.cuda()
                move = torch.cat((origin, zero_padding), 3)
                move = move[:, :, :, -origin.shape[3]:]
                if method == "subtract":
                    oMinusM = move - candidate
                else:
                    oMinusM = torch.cat((move, candidate), 1)
            oMinusM_List.append(oMinusM)
    elif position == "right":
        for disparity in range(D // 2**k):
            if disparity == 0:
                if method == "subtract":
                    oMinusM = origin - candidate
                else:
                    oMinusM = torch.cat((origin, candidate), 1)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding.cuda()
                move = torch.cat((zero_padding, origin), 3)
                move = move[:, :, :, :origin.shape[3]]
                if method == "subtract":
                    oMinusM = move - candidate
                else:
                    oMinusM = torch.cat((move, candidate), 1)
            oMinusM_List.append(oMinusM)
    cost_volume = torch.stack(oMinusM_List, dim=1)
    return cost_volume
