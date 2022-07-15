from util.loss import *
from pytorch_ssim import SSIM as SSIM

import torch
import torch.nn as nn



thr = 0.000001

def gradient_LIME(input):
    W = 1
    Gx = gradient(input, 'x', use_abs=True)
    Gy = gradient(input, 'y', use_abs=True)
    Gx = Gx * W
    Gy = Gy * W
    return Gx, Gy


def Loss_gradient_LIME(input, k=1):
    Gx, Gy = gradient_LIME(input)
    return k * (torch.mean(Gx) + torch.mean(Gy))


def gloss_R(R):
    R1 = R[:, 0:1, :, :]
    R2 = R[:, 1:2, :, :]
    R3 = R[:, 2:3, :, :]
    loss = Loss_gradient_LIME(R1, k=0.5) + Loss_gradient_LIME(R2, k=0.5) + Loss_gradient_LIME(R3, k=0.5)
    return loss


def tensor_gray_YUV(input):
    # return input[:, 0:1, :, :] * 0.299 + input[:, 1:2, :, :] * 0.587 + input[:, 2:3, :, :] * 0.114
    return input[:, 0:1, :, :] * 0.333 + input[:, 1:2, :, :] * 0.333 + input[:, 2:3, :, :] * 0.333


class restore_loss(nn.Module):
    def __init__(self):
        super(restore_loss, self).__init__()
        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()
        self.ssim = SSIM()

    def forward(self, R_list, L_list, I):
        loss_l2 = 0
        n = len(R_list)
        for i in range(n - 1):
            loss_l2 = loss_l2 + self.L2loss(R_list[i], R_list[i + 1]) + self.L2loss(L_list[i], L_list[i + 1])
        loss_l = torch.mean(gradient_LIME(L_list[n - 1], 'x') + gradient(L_list[n - 1], 'y'))
        loss_r = (1 - self.ssim(R_list[n - 1], R_list[0])) + self.L2loss(R_list[n - 1], R_list[0])
        LOSS = loss_l2 + 0.2 * loss_l + loss_r
        return LOSS


