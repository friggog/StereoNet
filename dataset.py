import csv
import glob
import os
import re
import numpy as np
import torch
import torch.utils.data
import logging
from PIL import Image
from torchvision import datasets, transforms


class kitti15(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None, mode="train"):
        self.dataset_root = data_root
        self.transform = transform
        self.left_img_folder = os.path.join(self.dataset_root, "image_2")
        self.right_img_folder = os.path.join(self.dataset_root, "image_3")
        self.left_ground_truth_folder = os.path.join(self.dataset_root, "disp_noc_0")
        # self.right_ground_truth_folder = os.path.join(self.dataset_root, "disp_occ_1")

        # self.left_img_list = os.listdir(self.left_img_folder)
        # self.right_img_list = os.listdir(self.right_img_folder)

        self.left_img_files = sorted(glob.glob(os.path.join(self.left_img_folder, '*10.png')))
        self.right_img_files = sorted(glob.glob(os.path.join(self.right_img_folder, '*10.png')))
        self.left_gt_files = sorted(glob.glob(os.path.join(self.left_ground_truth_folder, '*.png')))

        if mode == "train":
            self.left_img_files = self.left_img_files[:160]
            self.right_img_files = self.right_img_files[:160]
            self.left_gt_files = self.left_gt_files[:160]
        elif mode == "test":
            self.left_img_files = self.left_img_files[160:]
            self.right_img_files = self.right_img_files[160:]
            self.left_gt_files = self.left_gt_files[160:]

    def __getitem__(self, index):
        # print("index : %d" % index)
        # print(self.left_img_folder[000])
        left_img = Image.open("%s" % self.left_img_files[index])
        # print(left_img.size) # (1242, 375)
        # left_img.show()
        right_img = Image.open("%s" % self.right_img_files[index])
        left_gt = Image.open("%s" % self.left_gt_files[index])
        # left_gt.show()
        # print(left_gt.size)
        # print(np.asarray(left_gt).shape)
        # left_gt = self.transform(left_gt)

        # left_img = left_img.crop((0, 0, 370, 1238))
        # right_img = right_img.crop((0, 0, 370, 1238))
        left_gt = left_gt.crop((0, 0, 1234, 366))
        # left_gt.show()

        right_img = self.transform(np.asarray(right_img))
        left_img = self.transform(np.asarray(left_img))
        left_gt = torch.from_numpy(np.asarray(left_gt))

        # left_img = left_img.permute(1, 2, 0).float()
        # right_img = right_img.permute(1, 2, 0).float()
        left_gt = left_gt.unsqueeze(1)
        # print(left_gt.shape)
        left_gt = left_gt.permute(1, 0, 2).float()
        # print(left_gt.shape)

        # right_gt = self.transform(self.right_img_files[index])

        return left_img, right_img, left_gt

    def __len__(self):
        return len(self.left_img_files)


class sceneflow(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None, mode="train"):
        self.dataset_root = data_root
        self.transform = transform
        self.left_img_folder = os.path.join(self.dataset_root, "image_2")
        self.right_img_folder = os.path.join(self.dataset_root, "image_3")
        self.left_ground_truth_folder = os.path.join(self.dataset_root, "disp_noc_0")
        # self.right_ground_truth_folder = os.path.join(self.dataset_root, "disp_occ_1")

        # self.left_img_list = os.listdir(self.left_img_folder)
        # self.right_img_list = os.listdir(self.right_img_folder)

        self.left_img_files = sorted(glob.glob(os.path.join(self.left_img_folder, '*10.png')))
        self.right_img_files = sorted(glob.glob(os.path.join(self.right_img_folder, '*10.png')))
        self.left_gt_files = sorted(glob.glob(os.path.join(self.left_ground_truth_folder, '*.png')))

        if mode == "train":
            self.left_img_files = self.left_img_files[:160]
            self.right_img_files = self.right_img_files[:160]
            self.left_gt_files = self.left_gt_files[:160]
        elif mode == "test":
            self.left_img_files = self.left_img_files[160:]
            self.right_img_files = self.right_img_files[160:]
            self.left_gt_files = self.left_gt_files[160:]

    def __getitem__(self, index):
        # print("index : %d" % index)
        # print(self.left_img_folder[000])
        left_img = Image.open("%s" % self.left_img_files[index])
        # print(left_img.size) # (1242, 375)
        # left_img.show()
        right_img = Image.open("%s" % self.right_img_files[index])
        left_gt = Image.open("%s" % self.left_gt_files[index])
        # left_gt.show()
        # print(left_gt.size)
        # print(np.asarray(left_gt).shape)
        # left_gt = self.transform(left_gt)

        # left_img = left_img.crop((0, 0, 370, 1238))
        # right_img = right_img.crop((0, 0, 370, 1238))
        left_gt = left_gt.crop((0, 0, 1234, 366))
        # left_gt.show()

        right_img = self.transform(np.asarray(right_img))
        left_img = self.transform(np.asarray(left_img))
        left_gt = torch.from_numpy(np.asarray(left_gt))

        # left_img = left_img.permute(1, 2, 0).float()
        # right_img = right_img.permute(1, 2, 0).float()
        left_gt = left_gt.unsqueeze(1)
        # print(left_gt.shape)
        left_gt = left_gt.permute(1, 0, 2).float()
        # print(left_gt.shape)

        # right_gt = self.transform(self.right_img_files[index])

        return left_img, right_img, left_gt

    def __len__(self):
        return len(self.left_img_files)
