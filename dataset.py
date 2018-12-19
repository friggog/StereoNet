import csv
import glob
import os
import re
import numpy as np
import torch
import torch.utils.data
import trimesh
import logging


class kitti15(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        self.dataset_root = data_root
        self.transform = transform
        self.left_img_folder = os.path.join(self.dataset_root, "image_2")
        self.right_img_folder = os.path.join(self.dataset_root, "image_3")
        self.left_ground_truth_folder = os.path.join(self.dataset_root, "disp_occ_0")
        self.right_ground_truth_folder = os.path.join(self.dataset_root, "disp_occ_1")

        # self.left_img_list = os.listdir(self.left_img_folder)
        # self.right_img_list = os.listdir(self.right_img_folder)

        self.left_img_files = sorted(glob.glob(os.path.join(self.left_img_folder, '*.png')))
        self.right_img_files = sorted(glob.glob(os.path.join(self.right_img_folder, '*.png')))
        self.left_gt_files = sorted(glob.glob(os.path.join(self.left_ground_truth_folder, '*.png')))
        self.right_gt_files = sorted(glob.glob(os.path.join(self.right_ground_truth_folder, '*.png')))

    def __getitem__(self, index):
        left_img = self.transform(self.left_img_folder[index])
        right_img = self.transform(self.right_img_files[index])
        left_gt = self.transform(self.left_gt_files[index])
        right_gt = self.transform(self.right_img_files[index])

        return left_img, right_img, left_gt, right_gt

