import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from model import StereoNet
import time
import os
import logging
import copy
from torch.utils.data import Dataset, DataLoader
from dataset import kitti15
import visdom
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA


# number of samples in each iteration
batchSize = 8

learning_rate = 1e-3


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
        # transforms.RandomSizedCrop((388, 1240)),
        transforms.ToPILImage(),
        # transforms.CenterCrop((370, 1238)),
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # data_root = "/datasets/data_scene_flow/training"
    # train_set = kitti15(data_root=data_root, transform=transform, mode="train")
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=2,
    #                                            drop_last=True)
    #
    # test_set = kitti15(data_root=data_root, transform=transform, mode="test")
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSize, shuffle=False, num_workers=2,
    #                                           drop_last=True)

    data_path = "/datasets/sceneflow"

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        data_path)

    train_loader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
        batch_size=batchSize, shuffle=True, num_workers=1, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=batchSize, shuffle=False, num_workers=4, drop_last=False)

    """ cost_volume_method 
            -- subtract : the origin approach in the StereoNet paper
            -- concat   : concatenateing the padding image and the other image
    """
    cost_volume_method = "subtract"
    net = StereoNet(batchSize, cost_volume_method)

    if os.path.exists('./log/scene-flow-state.pkl'):
        checkpoint = torch.load('./log/scene-flow-state.pkl')
        # print(checkpoint)
        net.load_state_dict(checkpoint)
        print("checkpoint loaded")
    # print(net)
    net = net.to('cuda')

    logger.info("{} paramerters in total".format(sum(x.numel() for x in net.parameters())))
    # logger.info("{} paramerters in the last layer".format(sum(x.numel() for x in net.out_layer.parameters())))

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9)

    """ An exponentially-decaying learning rate:
    https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1"""
    def get_learning_rate(epoch):
        initial_lr = 1e-3
        k = 0.01
        lr = initial_lr * pow(np.e, -k*epoch)
        return lr

    """ show batch tensor img """
    def show_tensor_img(tensor_img):
        print(tensor_img.shape)  # torch.Size([4, 1, 366, 1234])
        tensor_img = tensor_img[0]
        # print(tensor_img.shape)  # torch.Size([1, 366, 1234])

        unloader = transforms.ToPILImage()
        image = tensor_img.cpu().clone()
        # print(image.shape)  # torch.Size([1, 366, 1234])
        image = unloader(image)
        image.show()

    def test(imgL, imgR, disp_true):
        net.eval()
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = net(imgL, imgR)
            # show_tensor_img(output3)

        pred_disp = output3.data.cpu()

        # # computing 3-px error#
        # true_disp = disp_true
        # index = np.argwhere(true_disp > 0)
        # disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        #     true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
        # correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
        #             disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
        #         index[0][:], index[1][:], index[2][:]] * 0.05)
        # torch.cuda.empty_cache()
        #
        # return 1 - (float(torch.sum(correct)) / float(len(index[0])))

        diff = torch.abs(pred_disp-disp_true)
        shape = imgL.shape
        # print(shape)  # torch.Size([4, 3, 370, 1238])
        acc = torch.sum(diff < 3)
        acc = acc.item() / float(shape[2] * shape[3] * batchSize)

        return acc



    loss_log = []
    max_acc = 0
    max_epo = 0
    # x = torch.arange(15000)
    # vis = visdom.Visdom(env=u'stereonet', use_incoming_socket=False)
    # x = torch.arange(1, 30, 0.01)
    # y = torch.sin(x)
    # vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})

    for epoch in range(15000):
        lr = get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("learning rate = %f" % lr)
        net.train()

        total_loss = 0
        total_test_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        acc_total = 0
        EPE_total = 0
        for batch_idx, (left_img, right_img, left_gt) in enumerate(train_loader):
            print(len(train_loader))

            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()

            left_img, right_img, left_gt = left_img.to('cuda'), right_img.to('cuda'), left_gt.to('cuda')
            print(left_img.shape)
            l_prediction = net(left_img, right_img)
            # print(l_prediction.shape)
            # print(left_gt.shape)
            # print(left_gt)
            # loss = loss_fn(l_prediction, left_gt) + loss_fn(r_prediction, left_gt)

            loss = nn.functional.smooth_l1_loss(l_prediction, left_gt)

            # show_tensor_img(l_prediction)
            # show_tensor_img(left_gt)

            # if batch_idx == 1:
            #     return

            diff = torch.abs(l_prediction.data.cpu() - left_gt.data.cpu())
            # print(diff)
            shape = left_img.shape
            # print(shape)  # torch.Size([4, 3, 370, 1238])
            acc = torch.sum(diff < 3)
            acc = acc.item() / float(shape[2]*shape[3]*batchSize)
            acc_total += acc

            diff = np.asarray(diff)
            EPE = np.mean(diff)
            EPE_total += EPE

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss
            # total_correct += correct

            if batch_idx == 19:
                # print("Acc = %f  <ACC> = %f" % (acc, acc_total / (batch_idx + 1)))
                logger.info("Average ACC = %f" % (acc_total / (batch_idx + 1)))
                # logger.info("Average EPE = %f" % (EPE_total / (batch_idx + 1)))


                logger.info("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} time={:.2}+{:.2}".format(
                    epoch, batch_idx+1, len(train_loader),
                    loss, total_loss / (batch_idx + 1),
                    # correct / len(data), total_correct / len(data) / (batch_idx + 1),
                          time_after_load - time_before_load,
                          time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()

            avg_loss = total_loss / (batch_idx + 1)

        if epoch > 8 and epoch % 10 == 0:
            """ TEST at epoch end"""
            for batch_idx, (imgL, imgR, disp_L) in enumerate(test_loader):
                # test_loss = test(imgL, imgR, disp_L)
                # print('Iter %d 3-px error in val = %.3f' % (batch_idx, test_loss * 100))
                # total_test_loss += test_loss
                test_loss = test(imgL, imgR, disp_L)
                print('Iter %d 3-px ACC in val = %.3f' % (batch_idx, test_loss * 100))
                total_test_loss += test_loss

            print('epoch %d total 3-px ACC in val = %.3f' % (epoch, total_test_loss / len(test_loader) * 100))
            if total_test_loss / len(test_loader) * 100 > max_acc:
                max_acc = total_test_loss / len(test_loader) * 100
                max_epo = epoch
            print('MAX epoch %d total ACC error = %.3f' % (max_epo, max_acc))

        # loss_log.append(total_loss)

        # vis.line(X=epoch, Y=loss_log[epoch], win='StereoNet_Loss', update='append' if epoch > 0 else None)
        with open("log.txt", 'a') as f:
            output = "epoch = %d LOSS = %f\n" % (epoch, total_loss)
            f.writelines(output)

        torch.save(net.state_dict(), os.path.join(log_dir, "scene-flow-state.pkl"))


if __name__=='__main__':
    main()
