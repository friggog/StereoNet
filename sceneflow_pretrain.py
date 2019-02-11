import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import FileListLoaders
from dataloader import ImageLoaders
from models import *
from utils.bilinear_sampler import apply_disparity
import pytorch_ssim
import torchvision.transforms as transforms
from PIL import Image, ImageOps


def gradient_x(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx


def gradient_y(img):
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy


def get_disparity_smoothness(disp, input_img):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)
    image_gradients_x = gradient_x(input_img)
    image_gradients_y = gradient_y(input_img)
    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x),
                                      1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y),
                                      1, keepdim=True))
    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y
    smoothness_x = torch.nn.functional.pad(
        smoothness_x, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
    smoothness_y = torch.nn.functional.pad(
        smoothness_y, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')
    return smoothness_x + smoothness_y


def SSIM(x, y):
    ssim_loss = pytorch_ssim.SSIM()
    return torch.clamp(1 - ssim_loss(x, y) / 2, 0, 1)


class WrapperToStopWeirdStuffHappening:

    def train(self, imgL, imgR, dispL, dispR):
        self.model.train()
        imgL = Variable(torch.FloatTensor(imgL)).cuda()
        imgR = Variable(torch.FloatTensor(imgR)).cuda()
        dispL = Variable(torch.FloatTensor(dispL)).cuda()
        dispR = Variable(torch.FloatTensor(dispR)).cuda()

        dispPreL, dispPreR, dispEstL, dispEstR = self.model(imgL, imgR)

        # loss = 0
        # # calculate loss images
        # imgEstL = apply_disparity(imgR, -dispEstL * self.im_scale)
        # imgEstR = apply_disparity(imgL, dispEstR * self.im_scale)
        # right_to_left_disp = apply_disparity(dispEstR, -dispEstL * self.im_scale)
        # left_to_right_disp = apply_disparity(dispEstL, dispEstR * self.im_scale)

        # # image loss
        # l1_left = F.smooth_l1_loss(imgEstL, imgL)
        # l1_right = F.smooth_l1_loss(imgEstR, imgR)
        # ssim_left = SSIM(imgEstL, imgL)
        # ssim_right = SSIM(imgEstR, imgR)
        # im_loss_left = 0.85 * ssim_left + 0.15 * l1_left
        # im_loss_right = 0.85 * ssim_right + 0.15 * l1_right
        # im_loss = im_loss_left + im_loss_right
        # # disparity smoothness loss
        # disp_loss_left = torch.mean(torch.abs(get_disparity_smoothness(dispEstL, imgL)))
        # disp_loss_right = torch.mean(torch.abs(get_disparity_smoothness(dispEstR, imgR)))
        # disp_loss = disp_loss_left + disp_loss_right
        # # consistency loss
        # lr_loss_left = torch.mean(torch.abs(right_to_left_disp - dispEstL))
        # lr_loss_right = torch.mean(torch.abs(left_to_right_disp - dispEstR))
        # lr_loss = lr_loss_left + lr_loss_right
        # # total and backprop
        # loss += im_loss + 0.1 * disp_loss + lr_loss

        # tgrey = transforms.ToPILImage(mode='L')
        # trgb = transforms.ToPILImage(mode='RGB')
        # il = trgb(imgL[0].data.cpu())
        # dl = tgrey(dispEstL[0].data.cpu())
        # dtl = tgrey(dispL[0].data.cpu().unsqueeze(0) / self.im_scale)
        # il.show()
        # dl.show()
        # dtl.show()
        # input()

        loss = 0
        # sum losses for L/R unrefined and refined predictions
        for outputL, outputR in [(dispPreL, dispPreR), (dispEstL, dispEstR)]:
            outputR = torch.squeeze(outputR, 1)
            outputL = torch.squeeze(outputL, 1)
            lossR = F.smooth_l1_loss(outputR * self.im_scale, dispR, reduction='mean')
            lossL = F.smooth_l1_loss(outputL * self.im_scale, dispL, reduction='mean')
            loss += (lossR + lossL)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()

    def test(self, imgL, imgR, dispL, dispR):
        self.model.eval()
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        imgL, imgR = imgL.cuda(), imgR.cuda()
        imgL = imgL * 0.5 + 0.5
        imgR = imgR * 0.5 + 0.5
        with torch.no_grad():
            _, _, outputL, outputR = self.model(imgL, imgR)
        estL = apply_disparity(imgR, -outputL.cuda() * self.im_scale)
        estR = apply_disparity(imgL, outputR.cuda() * self.im_scale)
        tgrey = transforms.ToPILImage(mode='L')
        trgb = transforms.ToPILImage(mode='RGB')

        i = 0  # output one of each batch
        norm = dispL[i].max(0)[0]
        norm = norm.max(0)[0].data.cpu() / self.im_scale
        dl = tgrey(outputL[i].data.cpu() / norm)
        dr = tgrey(outputR[i].data.cpu() / norm)
        dlt = tgrey(dispL[i].unsqueeze(0) / self.im_scale / norm)
        drt = tgrey(dispR[i].unsqueeze(0) / self.im_scale / norm)
        ts = int(time.time() * 100000)
        os.mkdir('out/' + str(ts))
        dl.save('out/' + str(ts) + '/Lde.png')
        dr.save('out/' + str(ts) + '/Rde.png')
        dlt.save('out/' + str(ts) + '/Ldt.png')
        drt.save('out/' + str(ts) + '/Rdt.png')
        pl = trgb(imgL[i].data.cpu())
        pr = trgb(imgR[i].data.cpu())
        el = trgb(estL[i].data.cpu())
        er = trgb(estR[i].data.cpu())
        pl.save('out/' + str(ts) + '/Lt.png')
        pr.save('out/' + str(ts) + '/Rt.png')
        el.save('out/' + str(ts) + '/Le.png')
        er.save('out/' + str(ts) + '/Re.png')

        # outputL = torch.squeeze(outputL.data.cpu(), 1)
        # outputR = torch.squeeze(outputR.data.cpu(), 1)
        # loss = torch.mean(torch.abs((outputL*width)-dispL)) + \
        #     torch.mean(torch.abs((outputR*width)-dispR))  # end-point-error

        outputL = torch.squeeze(outputL.data.cpu(), 1)
        outputR = torch.squeeze(outputR.data.cpu(), 1)
        lossR = F.smooth_l1_loss(outputR * self.im_scale, dispR, reduction='mean')
        lossL = F.smooth_l1_loss(outputL * self.im_scale, dispL, reduction='mean')
        loss = (lossR + lossL) / 2

        EPE = loss
        return EPE

    def main(self, TRAIN):
        if TRAIN:
            start_full_time = time.time()
            for epoch in range(self.epoch_start, self.epochs +1):
                print('This is epoch %d' % (epoch))
                self.scheduler.step()
                print("learning rate : %f " % self.scheduler.get_lr()[0])
                avg_train_loss = 0

                ## training ##
                for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, disp_crop_R) in enumerate(self.TrainImgLoader):
                    start_time = time.time()
                    loss = self.train(imgL_crop, imgR_crop,
                                      disp_crop_L, disp_crop_R)
                    avg_train_loss += loss
                    if batch_idx % 100 == 0:
                        print('Iter %d ave training loss = %.3f , time = %.2f' %
                              (batch_idx, loss, time.time() - start_time))
                print('epoch %d total training loss = %.3f' %
                      (epoch, avg_train_loss / len(self.TrainImgLoader)))

                # SAVE
                torch.save({
                    'state_dict': self.model.state_dict(),
                    'epoch': epoch + 1,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.checkpoint_path)
            print('full training time = %.2f HR' %
                  ((time.time() - start_full_time) /3600))

        total_test_loss = 0
        for batch_idx, (imgL, imgR, disp_L, disp_R) in enumerate(self.TestImgLoader):
            EPE = self.test(imgL, imgR, disp_L, disp_R)
            print('Iter %d EPE = %.3f' % (batch_idx, EPE))
            total_test_loss += EPE
        print('Total test loss = %.3f' %
              (total_test_loss /len(self.TestImgLoader)))

    def __init__(self, epochs, batch_size, load, start_epoch):
        root_path = "D:/StereoNet"
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)

        all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = FileListLoaders.SceneFlowList(
            './datasets/sceneflow/')

        batchSize = batch_size
        self.epochs = epochs
        self.im_scale = 960 * 0.2

        self.TrainImgLoader = torch.utils.data.DataLoader(
            ImageLoaders.SceneFlowImageLoader(all_left_img, all_right_img,
                                              all_left_disp, all_right_disp, True),
            batch_size=batchSize, shuffle=True, num_workers=12, drop_last=False)

        self.TestImgLoader = torch.utils.data.DataLoader(
            ImageLoaders.SceneFlowImageLoader(test_left_img, test_right_img,
                                              test_left_disp, test_right_disp, False),
            batch_size=batchSize, shuffle=False, num_workers=4, drop_last=False)

        cost_volume_method = "subtract"
        self.model = stereonet(batch_size=batchSize,
                               cost_volume_method=cost_volume_method)
        print("-- model using stereonet --")

        self.model = nn.DataParallel(self.model)
        self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.epoch_start = 0

        # if args.loadmodel is not None:
        self.checkpoint_path = root_path + "/checkpoints/sceneflow_checkpoint.tar"
        if os.path.exists(self.checkpoint_path) and load:
            state_dict = torch.load(self.checkpoint_path)
            self.model.load_state_dict(state_dict['state_dict'])
            if start_epoch == -1:
                self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.epoch_start = state_dict['epoch']
            print("-- checkpoint loaded --")

        if start_epoch >= 0:
            self.epoch_start = start_epoch

        if self.epoch_start > 0:
            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizer, gamma=0.9, last_epoch=self.epoch_start - 1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1, last_epoch=self.epoch_start -1)
        else:
            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizer, gamma=0.9)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)

        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', dest='load', action='store_true')
    parser.add_argument('--test-only', dest='skip_training', action='store_true')
    parser.add_argument('--epochs', dest='epochs', action='store', default=32, type=int)
    parser.add_argument('--start-at', dest='start_at', action='store', default=-1, type=int)
    parser.add_argument('--batch-size', dest='batch_size', action='store', default=1, type=int)
    args = parser.parse_args()
    a = WrapperToStopWeirdStuffHappening(args.epochs, args.batch_size, args.load, args.start_at)
    a.main(not args.skip_training)
