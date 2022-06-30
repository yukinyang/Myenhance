import torch
import torch.nn as nn

from util.util import *


def gradient(input, axis, use_abs=True):
    # k = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32)
    # k_x = torch.reshape(k, [1, 1, 2, 2])
    k = torch.tensor([[-1, 1]], dtype=torch.float32)
    k_x = torch.reshape(k, [1, 1, 1, 2])
    k_y = torch.transpose(k_x, 2, 3)
    # print(k_x.shape)
    # print(k_y.shape)
    size = (2, 2)
    if axis == 'x':
        kernel = k_x
        size = (1, 2)
    elif axis == 'y':
        kernel = k_y
        size = (2, 1)
    # print(kernel)
    conv = nn.Conv2d(1, 1, size, stride=1, bias=False)
    conv.weight.data = kernel
    conv.cuda()
    # print(size)
    # print(conv.weight.shape)
    if not use_abs:
        return conv(input)
    return torch.abs(conv(input))


def smooth(input_I, input_R):
    # s = input_I.shape
    # shape = [s[0], s[1], s[2], 1]
    input_R_1 = tensor_gray(input_R)
    input_R_2 = input_R_1.unsqueeze(1)
    # input_R = torch.reshape(input_R, shape)
    # print(input_I.shape)
    # print(input_R.shape)
    # loss1 = gradient(input_I, 'x') * torch.exp(-10 * gradient(input_I, 'x'))
    # loss2 = gradient(input_I, 'y') * torch.exp(-10 * gradient(input_I, 'y'))
    loss3 = gradient(input_R_2, 'x') * torch.exp(-10 * gradient(input_R_2, 'x'))
    loss4 = gradient(input_R_2, 'y') * torch.exp(-10 * gradient(input_R_2, 'y'))
    # return torch.mean(loss1) + torch.mean(loss2) + torch.mean(loss3) + torch.mean(loss4)
    # loss = torch.mean(loss1)
    return torch.mean(loss3) + torch.mean(loss4)

def smooth_R(input_R):
    input_R_1 = tensor_gray(input_R)
    input_R_2 = input_R_1.unsqueeze(1)
    loss1 = gradient(input_R_2, 'x') * torch.exp(-10 * gradient(input_R_2, 'x'))
    loss2 = gradient(input_R_2, 'y') * torch.exp(-10 * gradient(input_R_2, 'y'))
    return torch.mean(loss1) + torch.mean(loss2)

def sum_of_minpool(input):
    input = torch.abs(input)
    maxpool = nn.MaxPool2d((2, 2), stride=1)
    return torch.mean(-maxpool(-input))


def gradient_of_E_1(input):
    input = tensor_gray(input)
    input = input.unsqueeze(1)
    dx = gradient(input, 'x')
    dy = gradient(input, 'y')
    return torch.mean(torch.exp(-10 * dx)) + torch.mean(torch.exp(-10 * dy))


def gradient_of_E_2(input):
    input = tensor_gray(input)
    input = input.unsqueeze(1)
    dx = gradient(input, 'x', use_abs=False)
    dy = gradient(input, 'y', use_abs=False)
    ddxx = gradient(dx, 'x')
    ddxy = gradient(dx, 'y')
    ddyx = gradient(dy, 'x')
    ddyy = gradient(dy, 'y')
    loss1 = torch.mean(torch.exp(-10 * ddxx))
    loss2 = torch.mean(torch.exp(-10 * ddxy))
    loss3 = torch.mean(torch.exp(-10 * ddyx))
    loss4 = torch.mean(torch.exp(-10 * ddyy))
    return loss1 + loss2 + loss3 + loss4


def mean_illumination(input):
    return torch.mean(input)


class SCI_loss(nn.Module):
    def __init__(self):
        super(SCI_loss, self).__init__()
        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()

    def forward(self, in_list, R_list, L_list, stage):
        # stage = 2
        loss_R = self.L2loss(R_list[0], R_list[1]) + self.L2loss(R_list[0], in_list[0]) + self.L2loss(R_list[1], in_list[1])
        in0 = tensor_gray(in_list[0])
        in1 = tensor_gray(in_list[1])
        in0 = in0.unsqueeze(1)
        in1 = in1.unsqueeze(1)
        loss_L = self.L2loss(in0, L_list[0]) + self.L2loss(in1, L_list[1])
        loss_smooth = smooth_R(R_list[0]) + smooth_R(R_list[1])
        loss_in = self.L2loss(in_list[0], in_list[1])
        return 2 * loss_R + 2 * loss_L + 0.1 * loss_smooth + loss_in


class judge_loss(nn.Module):
    def __init__(self):
        super(judge_loss, self).__init__()
        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()

    def exposure_loss(self, new_L, L, R, thr=0.8):
        img = L * R
        target = torch.where(img > thr, R, L)
        return self.L2loss(new_L, target)

    def forward(self, new_L, L, R):
        loss_illumination = torch.abs(mean_illumination(new_L) - mean_illumination(L))
        R_1 = tensor_gray_3(R).unsqueeze(0)
        loss_L = self.L2loss(new_L, R_1)
        loss_exposure = self.exposure_loss(new_L, L, R)
        return loss_L + loss_illumination + loss_exposure





