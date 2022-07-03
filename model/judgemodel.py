from model.model import *
from model.SCImodel import *
from util.loss import *

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convs = self.make_convs()
        self.fc = self.make_fc()

    def make_convs(self):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
        ))
        for i in range(4):
            layers.append(nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ))
        return nn.Sequential(*layers)

    def make_fc(self):
        layers = []
        layers.append(nn.Sequential(
            nn.Linear(2080, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        ))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = input
        b = x.shape[0]
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Overexposure_net(nn.Module):
    def __init__(self):
        super(Overexposure_net, self).__init__()
        # self.Loss = judge_loss()
        self.conv_L = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
        self.conv_R = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
        self.out_convs = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, L, R):
        x1 = L
        x2 = R
        x1 = self.conv_L(x1)
        x2 = self.conv_R(x2)
        x = torch.cat([x1, x2], 1)
        new_L = self.out_convs(x)
        return new_L

    # def cal_loss(self, new_L, L, R):
    #     return judge_loss(new_L, L, R)


# class Overexposure_net_weight(nn.Module):
#     def __init__(self):
#         super(Overexposure_net_weight, self).__init__()
#         self.convs = self.make_convs()
#         self.loss = exposure_loss()
#
#     def make_convs(self):
#         layers = []
#         layers.append(nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, kernel_size=1),
#             nn.Sigmoid(),
#         ))
#         return nn.Sequential(*layers)
#
#     def forward(self, L, R, U, stage=3):
#         if stage == 0:
#             L_3 = torch.cat([L, L, L], 1)
#             img = L_3 * R
#             x = self.convs(img)
#             return L, x, img
#         L_list = []
#         x_list = []
#         img_list = []
#         for i in range(stage):
#             L_3 = torch.cat([L, L, L], 1)
#             img = L_3 * R
#             x = self.convs(img)
#             L_list.append(L)
#             x_list.append(x)
#             img_list.append(img)
#             L = L + x * U
#         return L_list, x_list, img_list
#
#     def cal_loss(self, L_list, x_list, img_list, stage=3):
#         return self.loss(L_list, x_list, img_list, stage)


class Overexposure_net_weight(nn.Module):
    def __init__(self):
        super(Overexposure_net_weight, self).__init__()
        self.convs = self.make_convs()
        self.loss = exposure_loss()

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

    def forward(self, L, R):
        L_3 = torch.cat([L, L, L], 1)
        img = L_3 * R
        img = torch.clamp(img, 0, 1)
        x = self.convs(img)
        return x, img


class illu_enhance(nn.Module):
    def __init__(self):
        super(illu_enhance, self).__init__()
        self.convs = self.make_convs()
        self.loss = exposure_loss()

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

    def forward(self, L, R):
        L_3 = torch.cat([L, L, L], 1)
        img = L_3 * R
        img = torch.clamp(img, 0, 1)
        x = self.convs(img)
        return x, img



