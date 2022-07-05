from util.loss import *

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


class LIMEloss(nn.Module):
    def __init__(self):
        super(LIMEloss, self).__init__()
        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()
        self.Avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

    def mean_loss(self, input, img=None):
        if img is None:
            loss = 1 - torch.mean(input)
        else:
            img_gray = tensor_gray(img)
            img_gray = img_gray.unsqueeze(1)
            # img_avg = self.Avgpool(img_gray)
            img_val = torch.mean(img_gray)
            input_val = torch.mean(input)
            k = 0.4
            # loss = (1 / (1 - torch.float_power((img_val), 10))) * ((k - img_val) / 2 * torch.float_power(input_val, 2) + 0.01 / (1 - input_val)) + 2
            # loss = torch.clamp(loss, 0, 5)
            if img_val > k:
                loss = 1 - input_val
            else:
                loss = 0
        return loss

    def gloss_R(self, R):
        R1 = R[:, 0:1, :, :]
        R2 = R[:, 1:2, :, :]
        R3 = R[:, 2:3, :, :]
        loss = Loss_gradient_LIME(R1, k=0.5) + Loss_gradient_LIME(R1, k=0.5) + Loss_gradient_LIME(R1, k=0.5)
        # R = tensor_gray(R)
        # R = R.unsqueeze(1)
        # loss = Loss_gradient_LIME(R, k=0.5)
        return loss
    
    def imgloss(self, R, L, img):
        L = torch.cat([L, L, L], 1)
        return self.L2loss(L * R, img)

    def forward(self, L_list, R_list, img=None):
        # L_list[0] is MAXc of IMG
        Loss_gragent = 0
        Loss_LMSE = 0
        Loss_mean = 0
        Loss_gradient_R = 0
        Loss_img = 0
        Loss_Rimg = 0
        n = len(L_list)
        for i in range(1, n):
            Loss_LMSE = Loss_LMSE + self.L2loss(L_list[i], L_list[i - 1])
            Loss_gragent = Loss_gragent + Loss_gradient_LIME(L_list[i], k=0.5)
            Loss_mean = Loss_mean + self.mean_loss(L_list[i], img)
            # Loss_img = Loss_img + self.imgloss(R_list[i - 1], L_list[i], img)
        # for i in range(0, n - 1):
        #     Loss_gradient_R = Loss_gradient_R + self.gloss_R(R_list[i])
        #     Loss_Rimg = Loss_Rimg + self.L2loss(R_list[i], img)
        # return Loss_gragent + Loss_LMSE + 2 * Loss_img + 0.2 * Loss_gradient_R
        return Loss_gragent + Loss_LMSE + 0.2 * Loss_mean










