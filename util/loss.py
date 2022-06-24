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
    loss1 = gradient(input_I, 'x') * torch.exp(-10 * gradient(input_I, 'x'))
    loss2 = gradient(input_I, 'y') * torch.exp(-10 * gradient(input_I, 'y'))
    loss3 = gradient(input_R_2, 'x') * torch.exp(-10 * gradient(input_R_2, 'x'))
    loss4 = gradient(input_R_2, 'y') * torch.exp(-10 * gradient(input_R_2, 'y'))
    return torch.mean(loss1) + torch.mean(loss2) + torch.mean(loss3) + torch.mean(loss4)
    # loss = torch.mean(loss1)
    # return loss


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



