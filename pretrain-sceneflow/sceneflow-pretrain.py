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
import math
from dataloader import listflowfile as lt
from dataloader import SceneFlowLoader as DA
from models import *
from torch.optim import RMSprop

parser = argparse.ArgumentParser(description='StereoNet')
parser.add_argument('--maxdisp', type=int ,default=160,
                    help='maxium disparity')
parser.add_argument('--datapath', default='/datasets/sceneflow/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=22,
                    help='number of epochs to train')
# parser.add_argument('--loadmodel', default= None, help='load model')
parser.add_argument('--loadmodel', default='/checkpoints', help='load model')
parser.add_argument('--savemodel', default='/checkpoints',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

root_path = "/home/oliver/PycharmProjects/StereoNet"

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

batchSize = 16
TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True),
    batch_size=batchSize, shuffle= True, num_workers= 12, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False),
    batch_size= batchSize, shuffle= False, num_workers= 12, drop_last=False)


# cost_volume_method = "concat"
cost_volume_method = "subtract"
model = stereonet(batch_size=batchSize, cost_volume_method=cost_volume_method)
print("-- model using stereonet --")

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = RMSprop(model.parameters(), lr=1e-3, weight_decay=0.0001)
epoch_start = 0
total_train_loss_save = 0

# if args.loadmodel is not None:
checkpoint_path = root_path + "/checkpoints/checkpoint_sceneflow.tar"
if args.loadmodel is not None and os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    epoch_start = state_dict['epoch']
    total_train_loss_save = state_dict['total_train_loss']
    print("-- checkpoint loaded --")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=epoch_start)
else:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print("-- no checkpoint --")

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def train(imgL,imgR, disp_L):
    model.train()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    #---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    #----
    optimizer.zero_grad()

    output = model(imgL, imgR)
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

    #---------
    mask = disp_true < args.maxdisp
    #----

    with torch.no_grad():
        output3 = model(imgL,imgR)

    output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]

    if len(disp_true[mask])==0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error
        EPE = loss

    return EPE


def main():

    start_full_time = time.time()
    for epoch in range(epoch_start, args.epochs+1):
        print('This is %d-th epoch' %(epoch))
        scheduler.step()
        print("learning rate : %f " % scheduler.get_lr()[0])
        avg_train_loss = 0

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            if batch_idx % 20 == 0:
                print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            avg_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, avg_train_loss/len(TrainImgLoader)))

        #SAVE
        torch.save({
            'state_dict': model.state_dict(),
            'total_train_loss': avg_train_loss,
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        EPE = test(imgL,imgR, disp_L)
        print('Iter %d EPE = %.3f' %(batch_idx, EPE))
        total_test_loss += EPE

    print('average test EPE = %.3f' %(total_test_loss/len(TestImgLoader)))


if __name__ == '__main__':
    main()
