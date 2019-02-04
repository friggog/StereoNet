import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import dataloader.readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

IMGNET_NORM_STATS = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

BASE_NORM_STATS = {'mean': [0.5, 0.5, 0.5],
                   'std': [0.5, 0.5, 0.5]}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class BaseImageLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, right_disparity, training, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.disp_R = right_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __len__(self):
        return len(self.left)


class SceneFlowImageLoader(BaseImageLoader):
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        disp_R = self.disp_R[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        w, h = left_img.size
        if disp_L == '':
            dataL = np.zeros((1, h, w))
            dataR = np.zeros((1, h, w))
        else:
            dataL, scaleL = self.dploader(disp_L)
            dataR, scaleR = self.dploader(disp_R)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        dataR = np.ascontiguousarray(dataR, dtype=np.float32)

        if self.training:
            t = transforms.Compose([
                transforms.ColorJitter(0.3, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(BASE_NORM_STATS['mean'], BASE_NORM_STATS['std'])])

            th, tw = 512, 960

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]
            dataR = dataR[y1:y1 + th, x1:x1 + tw]
        else:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(BASE_NORM_STATS['mean'], BASE_NORM_STATS['std']),
            ])

        left_img = t(left_img)
        right_img = t(right_img)
        return left_img, right_img, dataL, dataR


class Stereo360ImageLoader(BaseImageLoader):
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        w, h = left_img.size

        if self.training:
            t = transforms.Compose([
                transforms.ColorJitter(0.3, 0.2, 0.2, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(BASE_NORM_STATS['mean'], BASE_NORM_STATS['std'])
            ])
            x1 = random.randint(0, w)
            left_img = PIL.ImageChops.offset(left_img, x1, 0)
            right_img = PIL.ImageChops.offset(right_img, x1, 0)
            tw = w
            left_img = left_img.crop((0, 0, tw, h))
            right_img = right_img.crop((0, 0, tw, h))
        else:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(BASE_NORM_STATS['mean'], BASE_NORM_STATS['std'])
            ])

        left_img = t(left_img)
        right_img = t(right_img)

        dataL = torch.zeros((1, h, w))
        dataR = torch.zeros((1, h, w))
        return left_img, right_img, dataL, dataR


class KittiImageLoader(BaseImageLoader):
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        disp_R = self.disp_R[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        w, h = left_img.size
        if disp_L == '':
            dataL = torch.zeros((1, h, w))
            dataR = torch.zeros((1, h, w))
        else:
            dataL = self.dploader(disp_L)
            dataR = self.dploader(disp_R)

        if self.training:
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            # dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            dataL = dataL[:, y1:y1 + th, x1:x1 + tw]
            # dataR = np.ascontiguousarray(dataR, dtype=np.float32)
            dataR = dataR[:, y1:y1 + th, x1:x1 + tw]

            if disp_L == '':
                t = transforms.Compose([
                    transforms.ColorJitter(0.3, 0.2, 0.2, 0.1),
                    transforms.RandomHorizontalFlip(),  # no disparity so can flip images directly
                    transforms.ToTensor(),
                    transforms.Normalize(BASE_NORM_STATS['mean'], BASE_NORM_STATS['std'])])
            else:
                t = transforms.Compose([
                    transforms.ColorJitter(0.3, 0.2, 0.2, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(BASE_NORM_STATS['mean'], BASE_NORM_STATS['std'])])
        else:
            left_img = left_img.crop((w -1232, h -368, w, h))
            right_img = right_img.crop((w -1232, h -368, w, h))

            # dataL = dataL.crop((w -1232, h -368, w, h))
            # dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            dataL = dataL[:, h -368:h, w -1232:w]
            # # dataR = dataR.crop((w -1232, h -368, w, h))
            # dataR = np.ascontiguousarray(dataR, dtype=np.float32)
            dataR = dataR[:, h -368:h, w -1232:w]

            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(BASE_NORM_STATS['mean'], BASE_NORM_STATS['std'])])

        left_img = t(left_img)
        right_img = t(right_img)

        return left_img, right_img, dataL, dataR
