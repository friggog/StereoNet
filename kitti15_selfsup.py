from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from torch.optim import RMSprop
from models import *
from utils.bilinear_sampler import apply_disparity
import pytorch_ssim
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from dataloader import FileListLoaders
from dataloader import ImageLoaders


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


class Wrapper():

    def train(self, imgL, imgR):
        self.model.train()
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        _, width = imgR.shape[2:]

        imgL, imgR= imgL.cuda(), imgR.cuda()

        self.optimizer.zero_grad()

        dispEstL, dispEstR = self.model(imgL, imgR)
        loss = 0
        # calculate loss images
        imgEstL = apply_disparity(imgR, -dispEstL * self.im_scale)
        imgEstR = apply_disparity(imgL, dispEstR * self.im_scale)
        right_to_left_disp = apply_disparity(dispEstR, -dispEstL * self.im_scale)
        left_to_right_disp = apply_disparity(dispEstL, dispEstR * self.im_scale)
        # image loss
        l1_left = F.smooth_l1_loss(imgEstL, imgL)
        l1_right = F.smooth_l1_loss(imgEstR, imgR)
        ssim_left = SSIM(imgEstL, imgL)
        ssim_right = SSIM(imgEstR, imgR)
        im_loss_left = 0.85 * ssim_left + 0.15 * l1_left
        im_loss_right = 0.85 * ssim_right + 0.15 * l1_right
        im_loss = im_loss_left + im_loss_right
        # disparity smoothness loss
        disp_loss_left = torch.mean(torch.abs(get_disparity_smoothness(dispEstL, imgL)))
        disp_loss_right = torch.mean(torch.abs(get_disparity_smoothness(dispEstR, imgR)))
        disp_loss = disp_loss_left + disp_loss_right
        # consistency loss
        lr_loss_left = torch.mean(torch.abs(right_to_left_disp - dispEstL))
        lr_loss_right = torch.mean(torch.abs(left_to_right_disp - dispEstR))
        lr_loss = lr_loss_left + lr_loss_right
        # total and backprop
        loss += im_loss + disp_loss + lr_loss

        loss.backward()
        self.optimizer.step()

        return loss.data.item()

    def test(self, imgL, imgR, dispL, dispR):
        self.model.eval()
        imgL = Variable(torch.FloatTensor(imgL)) * 0.5 + 0.5
        imgR = Variable(torch.FloatTensor(imgR)) * 0.5 + 0.5
        imgL, imgR = imgL.cuda(), imgR.cuda()
        with torch.no_grad():
            outputL, outputR = self.model(imgL, imgR)
        estL = apply_disparity(imgR, -outputL.cuda() * self.im_scale)
        estR = apply_disparity(imgL, outputR.cuda() * self.im_scale)
        tgrey = transforms.ToPILImage(mode='L')
        trgb = transforms.ToPILImage(mode='RGB')

        i = 1  # output first of each batch
        dl = tgrey(outputL[i].data.cpu())
        dr = tgrey(outputR[i].data.cpu())
        # dlt = tgrey(dispL[i].unsqueeze(0) / self.im_scale * 2)
        # drt = tgrey(dispR[i].unsqueeze(0) / self.im_scale * 2)
        ts = int(time.time() * 100000)
        os.mkdir('out/' + str(ts))
        dl.save('out/' + str(ts) + '/Lde.png')
        dr.save('out/' + str(ts) + '/Rde.png')
        # dlt.save('out/' + str(ts) + '/Ldt.png')
        # drt.save('out/' + str(ts) + '/Rdt.png')
        pl = trgb(imgL.data.cpu()[i, :, 4:, :])
        pr = trgb(imgR.data.cpu()[i, :, 4:, :])
        el = trgb(estL.data.cpu()[i, :, 4:, :])
        er = trgb(estR.data.cpu()[i, :, 4:, :])
        pl.save('out/' + str(ts) + '/Lit.png')
        pr.save('out/' + str(ts) + '/Rit.png')
        el.save('out/' + str(ts) + '/Lie.png')
        er.save('out/' + str(ts) + '/Rie.png')

        outputL = torch.squeeze(outputL.data.cpu(), 1)
        outputR = torch.squeeze(outputR.data.cpu(), 1)
        # outputL = outputL[dispL.squeeze(1) > 0]
        # outputR = outputR[dispR.squeeze(1) > 0]
        # dispL = dispL[dispL.squeeze(1) > 0]
        # dispR = dispR[dispR.squeeze(1) > 0]
        lossR = F.smooth_l1_loss(outputR * self.im_scale, dispR, reduction='mean')
        lossL = F.smooth_l1_loss(outputL * self.im_scale, dispL, reduction='mean')
        loss = (lossR + lossL) / 2
        # loss = torch.mean(torch.abs((outputL * self.im_scale) - dispL)) + \
        #     torch.mean(torch.abs((outputR * self.im_scale) - dispR))

        EPE = loss
        return EPE

    # def test(self, imgL, imgR, dispL, dispR):
    #     self.model.eval()
    #     imgL = Variable(torch.FloatTensor(imgL))
    #     imgR = Variable(torch.FloatTensor(imgR))
    #     _, width = imgR.shape[2:]
    #     imgL, imgR = imgL.cuda(), imgR.cuda()

    #     with torch.no_grad():
    #         dispEstL, dispEstR = self.model(imgL, imgR)

    #     dispEstL = dispEstL * width
    #     dispEstR = dispEstR * width

    #     # computing 3-px error rate#
    #     index = np.argwhere(dispL > 0)
    #     aL[index[0][:], index[1][:], index[2][:]] = np.abs(
    #         dispEstL[index[0][:], index[1][:], index[2][:]] -dispL[index[0][:], index[1][:], index[2][:]])
    #     correctL = (aL[index[0][:], index[1][:], index[2][:]] < 3) | (
    #         aL[index[0][:], index[1][:], index[2][:]] < aL[index[0][:], index[1][:], index[2][:]] *0.05)
    #     torch.cuda.empty_cache()

    #     three_pixel_error_rate_L = 1 - \
    #         (float(torch.sum(correct)) /float(len(index[0])))

    #     index = np.argwhere(dispR > 0)
    #     aR[index[0][:], index[1][:], index[2][:]] = np.abs(
    #         dispEstR[index[0][:], index[1][:], index[2][:]] -dispR[index[0][:], index[1][:], index[2][:]])
    #     correctR = (aR[index[0][:], index[1][:], index[2][:]] < 3) | (
    #         aR[index[0][:], index[1][:], index[2][:]] < aR[index[0][:], index[1][:], index[2][:]] *0.05)
    #     torch.cuda.empty_cache()

    #     three_pixel_error_rate_R = 1 - \
    #         (float(torch.sum(correct)) /float(len(index[0])))

    #     return (three_pixel_error_rate_L + three_pixel_error_rate_R) / 2

    def main(self, batch_size, epochs, checkpoint_name, start_at, test_only):
        root_path = "D:/StereoNet"

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        self.im_scale = 1232 * 0.2

        all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = FileListLoaders.KittiList(
            './datasets/kitti15/')

        TrainImgLoader = torch.utils.data.DataLoader(
            ImageLoaders.KittiImageLoader(all_left_img, all_right_img,
                                          all_right_disp, all_left_disp, True),
            batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True)

        TestImgLoader = torch.utils.data.DataLoader(
            ImageLoaders.KittiImageLoader(test_left_img, test_right_img,
                                          test_left_disp, all_right_disp, False),
            batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

        cost_volume_method = "subtract"
        self.model = stereonet(batch_size=batch_size,
                               cost_volume_method=cost_volume_method)
        print("-- model using stereonet --")

        self.model = nn.DataParallel(self.model)
        self.model.cuda()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))

        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        epoch_start = 0
        max_acc = 0
        max_epo = 0
        checkpoint_path = root_path + "/checkpoints/" + checkpoint_name
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(state_dict['state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            epoch_start = state_dict['epoch']
            print("-- checkpoint loaded --")

        if start_at >= 0:
            epoch_start = start_at

        if epoch_start > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1, last_epoch=epoch_start -1)
            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizer, gamma=0.9, last_epoch=epoch_start -1)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)
            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizer, gamma=0.9)

        if not test_only:
            start_full_time = time.time()

            for epoch in range(epoch_start, epochs +1):
                print('This is %d-th epoch' % epoch)
                self.scheduler.step()
                print("learning rate : %f " % self.scheduler.get_lr()[0])

                total_train_loss = 0
                total_test_three_pixel_error_rate = 0

                ## training ##
                for batch_idx, (imgL_crop, imgR_crop, _, _) in enumerate(TrainImgLoader):
                    loss = self.train(imgL_crop, imgR_crop)
                    total_train_loss += loss
                    if batch_idx % 50 == 0:
                        print('=> Step %i loss %f' % (batch_idx, loss))

                print('Epoch %d average training loss = %.3f' %
                      (epoch, total_train_loss /len(TrainImgLoader)))

                ## Test ##
                # for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(TestImgLoader):
                #     test_three_pixel_error_rate = self.test(imgL, imgR, dispL, dispR)
                #     total_test_three_pixel_error_rate += test_three_pixel_error_rate

                # print('epoch %d total 3-px error in val = %.3f' %
                #       (epoch, total_test_three_pixel_error_rate /len(TestImgLoader) *100))

                # acc = (1 -total_test_three_pixel_error_rate /len(TestImgLoader)) *100
                # if acc > max_acc:
                # max_acc = acc
                # max_epo = epoch
                savefilename = root_path + '/checkpoints/checkpoint_finetune_kitti15.tar'
                torch.save({
                    'state_dict': self.model.state_dict(),
                    'total_train_loss': total_train_loss,
                    'epoch': epoch + 1,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'max_acc': max_acc,
                    # 'max_epoch': max_epo
                }, savefilename)
                # print("-- max acc checkpoint saved --")
                # print('MAX epoch %d test 3 pixel correct rate = %.3f' %
                # (max_epo, max_acc))

            print('full finetune time = %.2f HR' %
                  ((time.time() - start_full_time) /3600))

        total_test_loss = 0
        for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(TestImgLoader):
            loss = self.test(imgL, imgR, dispL, dispR)
            total_test_loss += loss
        print('Test loss %d' % total_test_loss / len(TestImgLoader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', dest='checkpoint_name', action='store', type=str, default="sceneflow_checkpoint.tar")
    parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=32)
    parser.add_argument('--batch_size', dest='batch_size', action='store', type=int, default=1)
    parser.add_argument('--start-at', dest='start_at', action='store', default=-1, type=int)
    parser.add_argument('--test-only', dest='test_only', action='store_true')
    args = parser.parse_args()
    a = Wrapper()
    a.main(args.batch_size, args.epochs, args.checkpoint_name, args.start_at, args.test_only)
