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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_epochs", type=int, default=80)
    parser.add_argument("--per_epochs", type=int, default=50)
    parser.add_argument("--per_samples", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_cpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--C", type=float, default=0.05)
    parser.add_argument("--data_path", type=str, default='../LOLdataset/imgs/')
    parser.add_argument("--img_size", type=int, default=[400, 600])
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--Epsilon", type=int, default=0.001)
    parser.add_argument("--decay_epoch", type=int, default=200)

    print(parser.parse_args())
    return parser.parse_args()


def LIMEtrain():
    opt = getparser()

    # model = LIME_decom(numlayers=3)
    model = RES_decom()
    denoise_model = D_net()
    # model = torch.nn.DataParallel(model)
    LOSS = RES_loss()
    LOSS_denoise = Denoise_loss()

    # checkpoint = torch.load('./save/100_LIME_decom.pth')
    # model.LIME.load_state_dict(checkpoint['LIME'])

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        model.cuda()
        LOSS.cuda()
        denoise_model.cuda()
        LOSS_denoise.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.9, 0.999)
    )
    optimizer_denoise = torch.optim.Adam(
        denoise_model.parameters(), lr=opt.lr, betas=(0.9, 0.999)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=LambdaLR(opt.epochs, 0, opt.epochs - 1).step
    )
    lr_scheduler_denoise = torch.optim.lr_scheduler.LambdaLR(
        optimizer_denoise, lr_lambda=LambdaLR(opt.epochs, 0, opt.epochs - 1).step
    )

    transforms_ = [
        transforms.Resize(int(opt.img_size[0] * 1.12), Image.BICUBIC),
        transforms.RandomCrop(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
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

    run_dir = get_dir_name('./run', 'Denoise')
    os.makedirs(run_dir)

    # checkpoint = torch.load('./save/100_RES_decom_lightnogray.pth')
    # model.load_state_dict(checkpoint['RES'])

    for epoch in range(0, opt.epochs):
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=numbatches)
        nowloss = 0
        denoiseloss = 0
        for i, batch in pbar:
            input = Variable(batch['img'].type(Tensor))
            sample_input = input

            # Train
            model.train()
            optimizer.zero_grad()

            input_list = []
            # L_list = []
            R_list = []
            R, L = model(input)

            R_list.append(R)
            # RR = R
            input_list.append(input)

            # Calculate loss
            Loss = LOSS(L, input_list, R_list)
            nowloss = nowloss + Loss

            Loss.backward()
            optimizer.step()

            ## denoise model train
            # denoise_model.train()
            optimizer_denoise.zero_grad()

            R_denoise = denoise_model(model(input))
            D_loss = smooth_R(R_denoise)
            # D_loss = nn.MSELoss(R, R_denoise)
            denoiseloss = denoiseloss + D_loss
            D_loss.backward()
            optimizer_denoise.step()

            now += 1

            if now % opt.per_samples == 0:
                ## enhance sample
                sample_single_img(now, sample_input[0, :, :, :], name='pre', dir=run_dir)
                sample_single_img(now + 1, sample_input[1, :, :, :], name='pre', dir=run_dir)
                R, L = model(input)
                R_denoise = denoise_model(R)
                sample_single_img(now, L[0, :, :, :], name='L', dir=run_dir)
                sample_single_img(now + 1, L[1, :, :, :], name='L', dir=run_dir)
                sample_single_img(now, R[0, :, :, :], name='R', dir=run_dir)
                sample_single_img(now + 1, R[1, :, :, :], name='R', dir=run_dir)
                sample_single_img(now, R_denoise[0, :, :, :], name='R_denoise', dir=run_dir)
                sample_single_img(now + 1, R_denoise[1, :, :, :], name='R_denoise', dir=run_dir)

        lr_scheduler.step()
        if (epoch >= (opt.save_epochs - 1) and (epoch + 1) % opt.per_epochs == 0):
            model_path = './save/' + str(epoch + 1) + '_RES_decom.pth'
            torch.save({'RES': model.state_dict()}, model_path)
            denoise_model_path = './save/' + str(epoch + 1) + '_Denoise.pth'
            torch.save({'Denoise': denoise_model.state_dict()}, denoise_model_path)
        nowloss = nowloss / numbatches
        denoiseloss = denoiseloss / numbatches
        print("epoch: " + str(epoch) + "   Loss: " + str(nowloss.cpu().detach().numpy()))
        print("epoch: " + str(epoch) + "   denoiseLoss: " + str(denoiseloss.cpu().detach().numpy()))
        print("======== epoch " + str(epoch) + " has been finished ========")



if __name__ == '__main__':
    LIMEtrain()






