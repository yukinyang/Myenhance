from model.model import *
from model.SCImodel import *
from util.loss import *

import torch
import torch.nn as nn


class RES_decom(nn.Module):
    def __init__(self):
        super(RES_decom, self).__init__()
        self.range = 3
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.inconv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.convs = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, stride=1),
        )

    def forward(self, input):
        x = input
        x = self.inconv(x)
        x = x + self.convs(x)
        x = self.outconv(x)
        '''  x should not be shaped in (0, 1], x can be greater  '''
        x = self.sigmoid(x)
        x = torch.clamp(x, 0.0001, self.range)
        L = x * self.range
        L = torch.cat([L, L, L], 1)
        R = input / L
        R = torch.clamp(R, 0, 1)
        return R, L


class GetU_net(nn.Module):
    def __init__(self):
        super(GetU_net, self).__init__()
        self.convs = self.make_convs()

    def make_convs(self):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        ))
        return nn.Sequential(*layers)

    def forward(self, input, k=0.05):
        x = input
        x = self.convs(x)
        x = x * k
        return x


class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        self.convs = self.make_convs()
        self.outconv = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def make_convs(self):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        ))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = input
        y = self.convs(x)
        z = self.outconv(y)
        return z


class LIME_decom(nn.Module):
    def __init__(self):
        super(LIME_decom, self).__init__()
        self.inconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.convs = self.make_convs(3)
        self.outconv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def make_convs(self, numlayers=3):
        layers = []
        for i in range(numlayers):
            layers.append(nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = input
        x = self.inconv(x)
        x = self.convs(x)
        x = self.outconv(x)
        # R = x[:, 0:3, :, :]
        # L = x[:, 3:4, :, :]
        return x


class Mynetwork(nn.Module):
    def __init__(self):
        super(Mynetwork, self).__init__()
        self.e = 0.001
        self.GetU = GetU_net()
        self.Denoise = Denoise_net()
        self.LIME = LIME_decom()

    def forward(self, input):
        x = input
        L = self.LIME(x)
        L = torch.clamp(L, self.e, 1)
        L_3 = torch.cat([L, L, L], 1)
        R = input / L_3
        R = torch.clamp(R, 0, 1)

        U = self.GetU(x)
        nL = L + U
        nR = self.Denoise(R)
        return R, L, U, nR, nL






