from model.model import *
from model.SCImodel import *
from util.loss import *

import torch
import torch.nn as nn


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









