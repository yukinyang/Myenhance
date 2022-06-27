from model.model import *
from util.loss import *

import torch
import torch.nn as nn


class enhance_net(nn.Module):
    def __init__(self, layers, range=0.1):
        super(enhance_net, self).__init__()
        self.layers = layers
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
        for i in range(layers):
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
        x = torch.clamp(x, 0, range)
        return x



class Network(nn.Module):
    def __init__(self, enhance=True, layers=3, stage=3):
        super(Network, self).__init__()

        self.stage = stage
        self.enhance = enhance
        self.Loss = SCI_loss()
        self.decom = KD_decom()
        self.enhance_net = enhance_net(layers)

    def forward(self, input):
        x = input
        in_list = []
        R_list = []
        L_list = []
        U_list = []
        for i in range(self.stage):
            in_list.append(x)
            R, L = self.decom(x)
            U = enhance_net(R)
            R_list.append(R)
            L_list.append(L)
            U_list.append(U)
            U = torch.cat([U, U, U], 1)
            if self.enhance:
                x = x + R * U
            else:
                x = x - R * U
        return in_list, R_list, L_list, U_list

    def cal_loss(self, in_list, R_list, L_list):
        loss = self.Loss(in_list, R_list, L_list, self.stage)
        return loss












