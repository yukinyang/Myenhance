from model.model import *
from model.judgemodel import *
from util.loss import *

import torch
import torch.nn as nn


class enhance_net(nn.Module):
    def __init__(self, layers=3, range=0.1):
        super(enhance_net, self).__init__()
        self.layers = layers
        self.range = range
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.convs = self.make_convs(self.layers)
        self.out_convs = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def make_convs(self, num):
        layers = []
        for i in range(0, 3):
            convs = []
            convs.append(nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU()
            ))
            layers.append(*convs)
        return nn.ModuleList(layers)

    def forward(self, input):
        x = input
        x = self.in_conv(x)
        for conv in self.convs:
            x = x + conv(x)
        x = self.out_convs(x)
        # x = torch.clamp(x, 0, self.range)
        x = x / (1 / self.range)
        return x



class Network(nn.Module):
    def __init__(self, enhance=True, layers=3, stage=2, weights=None):
        super(Network, self).__init__()

        self.stage = stage
        self.enhance = enhance
        self.Loss = SCI_loss()
        self.decom = KD_decom()
        self.enhance_net = enhance_net(layers)
        # self.exposure = Overexposure_net()

    def forward(self, input):
        x = input
        in_list = []
        R_list = []
        L_list = []
        U_list = []
        nL_list = []
        # R, L, E = self.decom(x)
        # return R,L,E
        for i in range(self.stage):
            in_list.append(x)
            R, L = self.decom(x)
            U = self.enhance_net(R)
            # print(U.shape)
            R_list.append(R)
            L_list.append(L)
            U_list.append(U)
            L = L + U
            nL_3 = torch.cat([L, L, L], 1)

            # nL = self.exposure(L + U, R)
            # nL_3 = torch.cat([nL, nL, nL], 1)
            # nL_list.append(nL)

            # U = torch.cat([U, U, U], 1)
            if self.enhance:
                # x = x + R * U
                x = R * nL_3
            else:
                x = x - R * U
        return in_list, R_list, L_list, U_list, nL_list

    def cal_loss(self, in_list, R_list, L_list, nL_list):
        loss = self.Loss(in_list, R_list, L_list, nL_list)
        return loss


class Testnet(nn.Module):
    def __init__(self):
        super(Testnet, self).__init__()
        self.decom = KD_decom()
        self.enhance_net = enhance_net()
        self.exposure = Overexposure_net_weight()

    def make_convs(self):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        ))
        return nn.Sequential(*layers)

    def forward(self, input, r):
        x = input
        R, L = self.decom(x)
        U = self.enhance_net(R)
        # U = 0.1 - U
        # L_list, x_list, img_list = self.exposure(L, R, U, r)
        # print('len(L_list)', len(L_list))
        for i in range(r):
            # x, img = self.exposure(L, R)
            # L = L + x * U
            L = L + U
            # L = self.exposure(L, R)
        # if len(L_list) == 1:
        #     L = L_list
        # else:
        #     L = L_list[-1]
        L = torch.cat([L, L, L], 1)
        newR = R * L
        newR = torch.clamp(newR, 0, 1)
        return newR, R









