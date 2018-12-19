import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.serialization import load_lua
import numpy as np
# from cStringIO import StringIO
from model import StereoNet
import cv2
import time
import os
import logging
import copy
from torch.utils.data import Dataset, DataLoader
from dataset import kitti15

# maxmium number of epochs to train the model
max_epoch = 5

# number of iterations in each epoch
iter_per_epoch = 50

# number of samples in each iteration
batchSize = 32

# gpu option. set 1 if available, else 0
gpu = 1

learning_rate = 0.01

h=256
w=512
maxdisp=192
batch=4


def main():
    arguments = copy.deepcopy(locals())

    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    data_root = "/datasets/data_scene_flow/training"
    train_set = kitti15(data_root=data_root, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=4, drop_last=True)

    net = StereoNet()
    print(net)
    net.cuda()

    logger.info("{} paramerters in total".format(sum(x.numel() for x in net.parameters())))
    logger.info("{} paramerters in the last layer".format(sum(x.numel() for x in net.out_layer.parameters())))

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9)
    loss_fn=nn.L1Loss()

    """ An exponentially-decaying learning rate:
    https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1"""
    def get_learning_rate(epoch):
        initial_lr = 1e-3
        k = 0.1
        lr = initial_lr * pow(np.e, k*epoch)
        return lr

    def train():
        for epoch in range(15000):
            lr = get_learning_rate(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            net.train()

            total_loss = 0
            total_correct = 0
            time_before_load = time.perf_counter()
            for batch_idx, (left_img, right_img, left_gt, right_gt) in enumerate(train_loader):
                time_after_load = time.perf_counter()
                time_before_step = time.perf_counter()

                left_img, right_img, left_gt, right_gt = left_img.cuda(), right_img.cuda(), left_gt.cuda(), right_gt.cuda()
                l_prediction, r_prediction = net(left_img, right_img)
                loss = loss_fn(l_prediction, left_gt) + loss_fn(r_prediction, right_gt)

                optimizer.zero_grad()
                loss.backward()
                # correct =

                total_loss += loss
                # total_correct += correct

                logger.info("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} time={:.2}+{:.2}".format(
                    epoch, batch_idx, len(train_loader),
                    loss, total_loss / (batch_idx + 1),
                    # correct / len(data), total_correct / len(data) / (batch_idx + 1),
                    time_after_load - time_before_load,
                    time.perf_counter() - time_before_step))
                time_before_load = time.perf_counter()

            torch.save(net.state_dict(), os.path.join(log_dir, "state.pkl"))

if __name__=='__main__':
    main()
