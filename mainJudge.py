from model.model import *
from util.util import *
from util.loss import *
from dataset.dataset import *
from model.SCImodel import *
from test import *
from model.judgemodel import *

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_cpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default='imgs/')
    parser.add_argument("--img_size", type=int, default=[300, 400])
    parser.add_argument("--decay_epoch", type=int, default=200)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--Decom_model_path", type=str, default='./save/100_SCI_model_KD_yuan.pth')
    parser.add_argument("--Enhance_model_path", type=str, default='./save/100_SCI_model_EN_yuan.pth')

    print(parser.parse_args())
    return parser.parse_args()


if __name__ == '__main__':
    opt = getparser()
    model0 = Testnet()
    model0.cuda()
    checkpoint_decom = torch.load(opt.Decom_model_path)
    checkpoint_enhance = torch.load(opt.Enhance_model_path)
    model0.decom.load_state_dict(checkpoint_decom['KD'])
    model0.enhance_net.load_state_dict(checkpoint_enhance['Enhance'])
    model0.eval()

    model = Overexposure_net_weight()

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        model.cuda()

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
    for epoch in range(0, opt.epochs):
        for i, batch in enumerate(dataloader):
            # set model input
            input = Variable(batch['img'].type(Tensor))

            # Train
            model.train()
            optimizer.zero_grad()

            R, L = model0.decom(input)
            U = model0.enhance_net(R)
            L_list, x_list, img_list = model(L, R, U, opt.stage)

            # Calculate loss
            Loss = model.cal_loss(L_list, x_list, img_list, opt.stage)
            # Loss = loss_l1
            nowloss = Loss
            Loss.backward()
            optimizer.step()
            now += 1

        lr_scheduler.step()
        if (epoch >= 49 and (epoch + 1) % 50 == 0) or epoch == 1:
            model_path = './save/' + str(epoch + 1) + '_model_EX.pth'
            torch.save({'Ex': model.state_dict()}, model_path)
        print("epoch: " + str(epoch) + "   Loss: " + str(nowloss.cpu().detach().numpy()))
        print("======== epoch " + str(epoch) + " has been finished ========")

    run_dir = get_dir_name('./run', 'EXtest')
    # os.makedirs(run_dir)
    # SCITest(run_dir)















