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


# number of samples in each iteration
batchSize = 8

learning_rate = 1e-3

h=256
w=512
maxdisp=192

# 禁止打印数组时使用省略号代替
np.set_printoptions(threshold=np.inf)
# 输出到log文件中
log = open("log/output.txt", "w")


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
        # transforms.ToPILImage(),
        # transforms.CenterCrop((370, 1238)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_root = "/datasets/data_scene_flow/training"
    train_set = kitti15(data_root=data_root, transform=transform, mode="train")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=2,
                                               drop_last=True)

    test_set = kitti15(data_root=data_root, transform=transform, mode="test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSize, shuffle=False, num_workers=2,
                                              drop_last=True)

    """ cost_volume_method 
            -- subtract : the origin approach in the StereoNet paper
            -- concat   : concatenateing the padding image and the other image
    """
    cost_volume_method = "subtract"
    net = StereoNet(batchSize, cost_volume_method)

    # print(net)
    net = net.to('cuda')

    logger.info("{} paramerters in total".format(sum(x.numel() for x in net.parameters())))
    # logger.info("{} paramerters in the last layer".format(sum(x.numel() for x in net.out_layer.parameters())))

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9)
    epoch = 0

    if os.path.exists('./log/state.pkl'):
        checkpoint = torch.load('./log/state.pkl')
        # print(checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("checkpoint loaded")

    """ An exponentially-decaying learning rate:
    https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1"""
    def get_learning_rate(epoch):
        initial_lr = 1e-4
        k = 0.01
        # lr = initial_lr * pow(np.e, -k*epoch)
        if epoch < 500:
            lr = 1e-6
        else:
            lr = 1e-7
        return lr

    """ show batch tensor img """
    def show_tensor_img(tensor_img):
        # print(tensor_img.shape)  # torch.Size([4, 1, 366, 1234])
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
        # print("output3.shape")
        # print(output3.shape)

        pred_disp = output3.data.cpu()

        diff = torch.abs(pred_disp-disp_true)

        shape = imgL.shape
        # print(shape)  # torch.Size([4, 3, 370, 1238])
        acc = torch.sum(diff < 1)
        acc = acc.item() / float(shape[2] * shape[3] * batchSize)
        # print(type(diff))
        # print("type(diff)")
        diff = np.asarray(diff)
        EPE = np.mean(diff)

        return acc, EPE

    """ Training """
    loss_log = []
    max_acc = 0
    max_epo = 0

    for epoch in range(epoch, 15000):
        lr = get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("learning rate = %f" % lr)
        net.train()

        total_loss = 0
        total_test_acc = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        acc_total = 0
        EPE_total = 0
        loss = 0
        for batch_idx, (left_img, right_img, left_gt) in enumerate(train_loader):

            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()

            left_img, right_img, left_gt = left_img.to('cuda'), right_img.to('cuda'), left_gt.to('cuda')
            l_prediction = net(left_img, right_img)

            # print(left_img.shape)
            # print(right_img.shape)

            # print("l_prediction.shape")
            # print(l_prediction.shape)  # torch.Size([8, 1, 366, 1234])

            loss = nn.functional.smooth_l1_loss(l_prediction, left_gt)

            diff = torch.abs(l_prediction.data.cpu() - left_gt.data.cpu())
            pred_disp_np = np.asarray(l_prediction.data.cpu())
            disp_true_np = np.asarray(left_gt.data.cpu())

            # if batch_idx == 0:

                # print(pred_disp_np, file=log)
                # print(pred_disp_np)
                # print(pred_disp_np.shape)
                # acc1 = np.sum(pred_disp_np > 100)

                # print(disp_true_np)

                # acc2 = np.sum(disp_true_np > 100)
                # print(acc1)
                # print(acc2)
                #
                # print("Out done!!!!!!!!")

            shape = left_img.shape

            acc = torch.sum(diff < 1)
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
                logger.info("Average EPE = %f" % (EPE_total / (batch_idx + 1)))

                logger.info("[{}:{}/{}] LOSS={:.3} <LOSS>={:.3} time={:.2}+{:.2}".format(
                    epoch, batch_idx+1, len(train_loader),
                    loss, total_loss / (batch_idx + 1),
                    # correct / len(data), total_correct / len(data) / (batch_idx + 1),
                          time_after_load - time_before_load,
                          time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()

            avg_loss = total_loss / (batch_idx + 1)

        """
        if epoch > 8 and epoch % 10 == 0:
            
            total_acc = 0
            total_EPE = 0
            for batch_idx, (imgL, imgR, disp_L) in enumerate(test_loader):
                # test_loss = test(imgL, imgR, disp_L)
                # print('Iter %d 3-px error in val = %.3f' % (batch_idx, test_loss * 100))
                # total_test_loss += test_loss
                test_acc, test_EPE = test(imgL, imgR, disp_L)
                print('Iter %d 1-px ACC in val = %.3f, EPE in val = %f' % (batch_idx, test_acc * 100, test_EPE))
                total_acc += test_acc
                total_EPE += test_EPE

            print('epoch %d total 1-px ACC in val = %.3f' % (epoch, total_test_acc / len(test_loader) * 100))
            if total_acc / len(test_loader) * 100 > max_acc:
                max_acc = total_test_acc / len(test_loader) * 100
                max_epo = epoch
            print('MAX epoch %d total ACC error = %.3f' % (max_epo, max_acc))
        """
        # loss_log.append(total_loss)

        # vis.line(X=epoch, Y=loss_log[epoch], win='StereoNet_Loss', update='append' if epoch > 0 else None)
        with open("log.txt", 'a') as f:
            output = "epoch = %d LOSS = %f\n" % (epoch, total_loss)
            f.writelines(output)

        """ SAVING & LOADING A GENERAL CHECKPOINT FOR INFERENCE AND/OR RESUMING TRAINING
            https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        """
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, os.path.join(log_dir, "state.pkl"))


if __name__=='__main__':
    main()
