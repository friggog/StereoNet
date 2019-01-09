from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
# 禁止打印数组时使用省略号代替
np.set_printoptions(threshold=np.inf)

from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA

from models import *

root_path = "/home/oliver/PycharmProjects/StereoNet"

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=160,
                    help='maxium disparity')
# parser.add_argument('--model', default='stackhourglass', help='select model')
parser.add_argument('--model', default='stereonet', help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/datasets/data_scene_flow/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')

parser.add_argument('--loadmodel', default=root_path+'/checkpoints/checkpoint_sceneflow.tar', help='load model')
# parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--savemodel', default=root_path+"/checkpoints",
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)


batchSize = 16
TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True),
    batch_size=batchSize, shuffle= True, num_workers= 12, drop_last=True)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False),
    batch_size=batchSize, shuffle= False, num_workers= 4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
elif args.model == 'stereonet':
    cost_volume_method = "subtract"
    model = stereonet(batch_size=batchSize, cost_volume_method=cost_volume_method)
    print("-- model using stereonet --")
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
epoch_start = 0
total_train_loss_save = 0

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

checkpoint_path = root_path + "/log/checkpoint_sceneflow.tar"
if args.loadmodel is not None and os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    epoch_start = state_dict['epoch']
    total_train_loss_save = state_dict['total_train_loss']
    print("-- checkpoint loaded --")

def train(imgL,imgR,disp_L):
    model.train()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    #---------
    mask = (disp_true > 0)
    mask.detach_()
    #----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL,imgR)
        output1 = torch.squeeze(output1,1)
        output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
    elif args.model == 'basic':
        output = model(imgL,imgR)
        output = torch.squeeze(output,1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)
    elif args.model == 'stereonet':
        output = model(imgL, imgR)
        # print("output.shape")
        # print(output.shape)  # torch.Size([4, 1, 256, 512])
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data.item()

def test(imgL,imgR,disp_true):
    model.eval()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL,imgR)
        output = torch.squeeze(output3.data.cpu(), 1)

    pred_disp = output

    # --------- From pretrain ---------------
    mask1 = disp_true < args.maxdisp

    mask2 = (disp_true > 0)
    mask2.detach_()

    EPE1 = torch.mean(torch.abs(output[mask1] - disp_true[mask1]))  # end-point-error
    EPE2 = torch.mean(torch.abs(output[mask2] - disp_true[mask2]))  # end-point-error

    EPE = torch.mean(torch.abs(output - disp_true))  # end-point-error
    print("EPE1")
    print(EPE1)
    print("EPE2")
    print(EPE2)
    print("EPE")
    print(EPE)
    # ---------------------------------------

    # computing 3-px error rate#
    true_disp = disp_true
    index = np.argwhere(true_disp>0)
    # print("index.shape")
    # print(index.shape)
    # print("disp_true.shape")
    # print(disp_true.shape)
    # print("pred_disp.shape")
    # print(pred_disp.shape)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)
    torch.cuda.empty_cache()
    three_pixel_error_rate = 1-(float(torch.sum(correct))/float(len(index[0])))

    return three_pixel_error_rate, EPE

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc=0
    max_epo=0
    start_full_time = time.time()

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        total_test_three_pixel_error_rate = 0
        total_test_EPE = 0
        adjust_learning_rate(optimizer,epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss

            # TODO Only debug using
            # for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            #     test_loss, test_EPE = test(imgL, imgR, disp_L)
            #     print('Iter %d 3-px error in val = %.3f, EPE = %.3f' % (batch_idx, test_loss * 100, test_EPE))

        print('epoch %d average training loss = %.3f' % (epoch, total_train_loss/len(TrainImgLoader)))

        ## Test ##
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_three_pixel_error_rate, test_EPE = test(imgL,imgR, disp_L)
            print('Iter %d 3-px error in val = %.3f, EPE = %.3f' %(batch_idx, test_three_pixel_error_rate*100, test_EPE))
            total_test_three_pixel_error_rate += test_three_pixel_error_rate
            total_test_EPE += test_EPE


        print('epoch %d total 3-px error in val = %.3f, EPE = %.3f' %(epoch, total_test_three_pixel_error_rate/len(TestImgLoader)*100, total_test_EPE/len(TestImgLoader)))

        acc = (1-total_test_three_pixel_error_rate/len(TestImgLoader))*100
        if acc > max_acc:
            max_acc = acc
            max_epo = epoch
        print('MAX epoch %d test 3 pixel correct rate = %.3f' %(max_epo, max_acc))

        savefilename = root_path + '/log/checkpoint_finetune_kitti15.tar'
        torch.save({
            'state_dict': model.state_dict(),
            # 'train_loss': total_train_loss/len(TrainImgLoader),
            'total_train_loss': total_train_loss,
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
        }, savefilename)
        print("-- checkpoint saved --")

    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
    main()
