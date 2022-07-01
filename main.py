from model.model import *
from util.util import *
from util.loss import *
from dataset.dataset import *
from test import *

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


'''
Loss
    Loss_l1 = nn.L1Loss()       一致性损失
    minpool(input)              最小池化和平均值
    graident(x), graident(y)    求xy方向梯度
    噪声损失函数
    Loss_E = minpool(input)
             + torch.exp(-torch.mean(graident(input, x)) - torch.mean(graident(input, y)))
             + torch.exp(-二阶梯度)
    RI平滑损失
    Loss_sm = loss.smooth()
'''


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_cpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default='../LOLdataset/imgs/')
    parser.add_argument("--img_size", type=int, default=[400, 600])
    parser.add_argument("--decay_epoch", type=int, default=200)

    print(parser.parse_args())
    return parser.parse_args()


if __name__ == '__main__':
    opt = getparser()

    Loss_l1 = nn.L1Loss()

    # model = Decom(layers=5)
    model = KD_decom_s()

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        model.cuda()
        Loss_l1.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.9, 0.999)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=LambdaLR(opt.epochs, 0, opt.epochs - 1).step
    )

    transforms_ = [
        transforms.Resize(int(opt.img_size[0] * 1.12), Image.BICUBIC),
        transforms.RandomCrop(opt.img_size),
        # transforms.Resize(opt.img_size, transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset(opt.data_path, transform_=transforms_),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpus,
    )
    print(len(dataloader))

    now = 0
    nowloss = 0

    run_dir = get_dir_name('./run', 'Decom_train')
    os.makedirs(run_dir)

    for epoch in range(0, opt.epochs):
        for i, batch in enumerate(dataloader):
            # set model input
            input = Variable(batch['img'].type(Tensor))
            # print(input.shape)

            # Train
            model.train()
            optimizer.zero_grad()

            R, L = model(input)

            # Calculate loss
            L3 = torch.concat([L, L, L], 1)
            # print(L3.shape)
            rec = R * L3
            loss_l1 = Loss_l1(rec, input)
            loss_l1_RL = Loss_l1(R * L3, input)
            loss_sm = smooth(L, R)
            # loss_minp_E = sum_of_minpool(E)
            # loss_d1_E = gradient_of_E_1(E)
            # loss_d2_E = gradient_of_E_2(E)

            # Loss = loss_l1 + 0.8 * loss_l1_RL + 0.1 * loss_sm + 0.1 * loss_minp_E + 0.05 * loss_d1_E + 0.05 * loss_d2_E
            Loss = loss_l1 + 0.8 * loss_l1_RL + 0.1 * loss_sm
            # Loss = loss_l1
            nowloss = Loss
            Loss.backward()
            optimizer.step()
            now += 1

            if now % 99 == 0:
                sample(R[0, :, :, :], L[0, :, :, :], now, input[0, :, :, :], run_dir)
        lr_scheduler.step()
        if (epoch >= 99 and (epoch + 1) % 50 == 0) or epoch == 1:
            model_path = './save/' + str(epoch + 1) + '_decom_s_LOLset.pth'
            torch.save({'KD':model.state_dict()}, model_path)
        print("epoch: " + str(epoch) + "   Loss: " + str(nowloss.cpu().detach().numpy()))
        print("======== epoch " + str(epoch) + " has been finished ========")

    # Test
    load_dir = './save/' + str(opt.epochs) + '_decom_LOLset.pth'
    Testdir = './testimg/'
    Savedir = get_dir_name('./run', 'Decom_test')
    os.makedirs(Savedir)
    Testmodel = KD_decom()
    Testmodel.cuda()
    DecomTest(Testmodel, Testdir, Savedir, load_dir)













