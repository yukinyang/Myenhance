from model.model import *
from model.SCImodel import *
from util.loss import *

import torch
import torch.nn as nn


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

    def forward(self, input, k=0.2):
        x = input
        x = self.convs(x)
        x = x * k
        return x


class Denoise_net(nn.Module):
    def __init__(self):
        super(Denoise_net, self).__init__()
        self.convs = self.make_convs()
        self.outconv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def make_convs(self):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        ))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = input
        return x


class LIME_decom(nn.Module):
    def __init__(self, numlayers):
        super(LIME_decom, self).__init__()
        self.inconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.convs = self.make_convs(numlayers)
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
    def __init__(self, numlayers):
        super(Mynetwork, self).__init__()
        self.e = 0.001
        self.GetU = GetU_net()
        self.Denoise = Denoise_net()
        self.LIME = LIME_decom(numlayers)

    def forward(self, input):
        x = input
        L = self.LIME(x)
        L = torch.clamp(L, self.e, 1)
        R = input / L
        R = torch.clamp(R, 0, 1)

        U = self.GetU(x)
        nL = L + U
        nR = self.Denoise(R)
        return R, L, U, nR, nL






