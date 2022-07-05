from model.model import *
from model.LIMEmodel import *
from util.util import *
from util.loss import *
from util.loss_LIME import *
from dataset.dataset import *
from test import *

import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_epochs", type=int, default=40)
    parser.add_argument("--per_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_cpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default='../LOLdataset/imgs/')
    parser.add_argument("--img_size", type=int, default=[400, 600])
    parser.add_argument("--stage", type=int, default=10)
    parser.add_argument("--Epsilon", type=int, default=0.001)
    parser.add_argument("--decay_epoch", type=int, default=200)

    print(parser.parse_args())
    return parser.parse_args()


def LIMEtrain():
    opt = getparser()

    # model = LIME_decom(numlayers=3)
    model = Mynetwork(numlayers=3)
    model = torch.nn.DataParallel(model)
    LOSS = LIMEloss()

    checkpoint = torch.load('./save/100_LIME_decom.pth')
    model.LIME.load_state_dict(checkpoint['LIME'])

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        model.cuda()
        LOSS.cuda()

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
    numbatches = len(dataloader)

    run_dir = get_dir_name('./run', 'MY_train')
    os.makedirs(run_dir)

    for epoch in range(0, opt.epochs):
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=numbatches)
        nowloss = 0
        for i, batch in pbar:
            input = Variable(batch['img'].type(Tensor))

            # Train
            model.train()
            optimizer.zero_grad()

            L_list = []
            R_list = []
            nR_list = []
            nL_list = []
            U_list = []
            input_list = []
            imgC_list = []

            input_list.append(input)
            imgC_list.append(MAXC(input))

            for i in range(opt.stage):
                imgC_list.append(MAXC(input))
                R, L, U, nR, nL = model(input)
                input = R * L
                input = torch.clamp(input, 0, 1)
                # print(R.shape, L.shape)
                # print(L.shape)
                L_list.append(L)
                R_list.append(R)
                nR_list.append(nR)
                nL_list.append(nL)
                U_list.append(U)
                input_list.append(input)

            # Calculate loss
            Loss = LOSS(L_list, R_list, input_list, U_list, nR_list, nL_list, imgC_list)
            nowloss = nowloss + Loss

            Loss.backward()
            optimizer.step()
            now += 1

            if now % 101 == 0:
                L = model(input)
                # save 2 groups
                L1 = L[0, :, :, :]
                sample_gray_img(now, L1, name='L', dir=run_dir)
                sample_single_img(now, input[0, :, :, :], name='input', dir=run_dir)
                R = input[0, :, :, :] / (L[0, :, :, :] + opt.Epsilon)
                R = torch.clamp(R, 0, 1)
                # R1 = R[0, :, :, :]
                sample_single_img(now, R, name='R', dir=run_dir)
                # L1_3 = torch.cat([L1, L1, L1], 0)
                # img1 = R1 * L1_3
                # sample_single_img(now, img1, name='rec', dir=run_dir)

                L2 = L[1, :, :, :]
                sample_gray_img(now + 1, L2, name='L', dir=run_dir)
                sample_single_img(now + 1, input[1, :, :, :], name='input', dir=run_dir)
                R = input[1, :, :, :] / (L[1, :, :, :] + opt.Epsilon)
                R = torch.clamp(R, 0, 1)
                # R2 = R[1, :, :, :]
                sample_single_img(now + 1, R, name='R', dir=run_dir)
                # L2_3 = torch.cat([L2, L2, L2], 0)
                # img2 = R2 * L2_3
                # sample_single_img(now + 1, img2, name='rec', dir=run_dir)

        lr_scheduler.step()
        if (epoch >= (opt.save_epochs - 1) and (epoch + 1) % opt.per_epochs == 0):
            model_path = './save/' + str(epoch + 1) + '_LIME_decom.pth'
            torch.save({'LIME': model.state_dict()}, model_path)
        print("epoch: " + str(epoch) + "   Loss: " + str(nowloss.cpu().detach().numpy()))
        print("======== epoch " + str(epoch) + " has been finished ========")



if __name__ == '__main__':
    LIMEtrain()






