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

from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA

from models import *

root_path = "/home/oliver/PycharmProjects/StereoNet"

parser = argparse.ArgumentParser(description='StereoNet')
parser.add_argument('--maxdisp', type=int ,default=160,
                    help='maxium disparity')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/datasets/data_scene_flow/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=2000,
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

cost_volume_method = "subtract"
# cost_volume_method = "concat"
model = stereonet(batch_size=batchSize, cost_volume_method=cost_volume_method)
print("-- model using stereonet --")

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
# optimizer = RMSprop(model.parameters(), lr=1e-3, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


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

    with torch.no_grad():
        output3 = model(imgL,imgR)
        output = torch.squeeze(output3.data.cpu(), 1)

    pred_disp = output

    # computing 3-px error rate#
    true_disp = disp_true
    index = np.argwhere(true_disp>0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)
    torch.cuda.empty_cache()

    three_pixel_error_rate = 1-(float(torch.sum(correct))/float(len(index[0])))

    return three_pixel_error_rate

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    print("lr = %f" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    epoch_start = 0
    max_acc = 0
    max_epo = 0
    checkpoint_path = root_path + "/checkpoints/checkpoint_finetune_kitti15.tar"
    if args.loadmodel is not None and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        epoch_start = state_dict['epoch']
        max_acc = state_dict['max_acc']
        max_epo = state_dict['max_epoch']
        print("-- checkpoint loaded --")

    start_full_time = time.time()

    for epoch in range(epoch_start, args.epochs+1):
        print('This is %d-th epoch' % epoch)
        # scheduler.step()
        # print("learning rate : %f " % scheduler.get_lr()[0])
        adjust_learning_rate(optimizer, epoch)

        total_train_loss = 0
        total_test_three_pixel_error_rate = 0

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            total_train_loss += loss

        print('epoch %d average training loss = %.3f' % (epoch, total_train_loss/len(TrainImgLoader)))

        ## Test ##
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_three_pixel_error_rate = test(imgL,imgR, disp_L)
            total_test_three_pixel_error_rate += test_three_pixel_error_rate

        print('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_three_pixel_error_rate/len(TestImgLoader)*100))

        acc = (1-total_test_three_pixel_error_rate/len(TestImgLoader))*100
        if acc > max_acc:
            max_acc = acc
            max_epo = epoch
            savefilename = root_path + '/checkpoints/checkpoint_finetune_kitti15.tar'
            torch.save({
                'state_dict': model.state_dict(),
                'total_train_loss': total_train_loss,
                'epoch': epoch + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'max_acc': max_acc,
                'max_epoch': max_epo
            }, savefilename)
            print("-- max acc checkpoint saved --")
        print('MAX epoch %d test 3 pixel correct rate = %.3f' %(max_epo, max_acc))

    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
    main()
