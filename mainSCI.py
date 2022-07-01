from model.model import *
from util.util import *
from util.loss import *
from dataset.dataset import *
from model.SCImodel import *
from test import *

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
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--n_cpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default='../LOLdataset/imgs/')
    parser.add_argument("--img_size", type=int, default=[400, 600])
    parser.add_argument("--decay_epoch", type=int, default=200)

    print(parser.parse_args())
    return parser.parse_args()


if __name__ == '__main__':
    opt = getparser()

    model = Network()
    Loss_l1 = nn.L1Loss()

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

    checkpoint = torch.load('./save/200_decom_s_LOLset.pth')
    model.decom.load_state_dict(checkpoint['KD'])

    now = 0
    nowloss = 0
    for epoch in range(0, opt.epochs):
        for i, batch in enumerate(dataloader):
            # set model input
            input = Variable(batch['img'].type(Tensor))
            # print(input.shape)

            # Train
            model.train()
            optimizer.zero_grad()

            in_list, R_list, L_list, U_list, nL_list = model(input)

            # Calculate loss
            Loss = model.cal_loss(in_list, R_list, L_list, nL_list)
            # Loss = loss_l1
            nowloss = Loss
            Loss.backward()
            optimizer.step()
            now += 1

            # if now % 40 == 0:
            #     sample(R[0, :, :, :], L[0, :, :, :], E[0, :, :, :], now, input[0, :, :, :])
        lr_scheduler.step()
        if (epoch >= 49 and (epoch + 1) % 50 == 0) or epoch == 1:
            model_KD_path = './save/' + str(epoch + 1) + '_SCI_model_KD.pth'
            model_enhance_path = './save/' + str(epoch + 1) + '_SCI_model_EN.pth'
            model_ex_path = './save/' + str(epoch + 1) + '_SCI_model_EX.pth'
            torch.save({'KD':model.decom.state_dict()}, model_KD_path)
            torch.save({'Enhance':model.enhance_net.state_dict()}, model_enhance_path)
            torch.save({'Ex':model.exposure.state_dict()}, model_ex_path)
        print("epoch: " + str(epoch) + "   Loss: " + str(nowloss.cpu().detach().numpy()))
        print("======== epoch " + str(epoch) + " has been finished ========")

    run_dir = get_dir_name('./run', 'SCItest')
    os.makedirs(run_dir)
    SCITest(run_dir)










