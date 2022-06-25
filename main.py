from model.model import *
from util.util import *
from util.loss import *
from dataset.dataset import *

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_cpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--data_path", type=str, default='imgs/')
    parser.add_argument("--img_size", type=int, default=[300, 400])
    parser.add_argument("--decay_epoch", type=int, default=800)

    print(parser.parse_args())
    return parser.parse_args()


if __name__ == '__main__':
    opt = getparser()

    Loss_l1 = nn.L1Loss()

    model = Decom(layers=5)

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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    for epoch in range(0, 300):
        for i, batch in enumerate(dataloader):
            # set model input
            input = Variable(batch['img'].type(Tensor))
            # print(input.shape)

            # Train
            model.train()
            optimizer.zero_grad()

            # R, L, E = model(input)
            x = model(input)
            R = x[:, 0:3, :, :]
            L = x[:, 3:4, :, :]
            E = x[:, 4:7, :, :] - 0.5
            # print(R.shape)

            # Calculate loss
            L3 = torch.concat([L, L, L], 1)
            # print(L3.shape)
            rec = R * L3 + E
            loss_l1 = Loss_l1(rec, input)
            loss_sm = smooth(L, R)
            loss_minp_E = sum_of_minpool(E)
            loss_d1_E = gradient_of_E_1(E)
            loss_d2_E = gradient_of_E_2(E)

            Loss = loss_l1 + 0.2 * loss_sm + 0.3 * loss_minp_E + 0.1 * loss_d1_E + 0.1 * loss_d2_E
            nowloss = Loss
            Loss.backward()
            optimizer.step()
            now += 1

            if now % 50 == 0:
                sample(R[0, :, :, :], L[0, :, :, :], E[0, :, :, :], now)
        lr_scheduler.step()
        print("epoch: " + str(epoch) + "   Loss: " + str(nowloss))
        print("======== epoch " + str(epoch) + " has been finished ========")












