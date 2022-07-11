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


def gloss_R(R):
    R1 = R[:, 0:1, :, :]
    R2 = R[:, 1:2, :, :]
    R3 = R[:, 2:3, :, :]
    loss = Loss_gradient_LIME(R1, k=0.5) + Loss_gradient_LIME(R2, k=0.5) + Loss_gradient_LIME(R3, k=0.5)
    return loss


def Light(L, img):
    loss = 0
    batch = img.shape[0]
    for i in range(batch):
        img_now = img[i:i + 1, :, :, :]
        L_now = L[i:i + 1, :, :, :]
        if torch.mean(img_now) > 0.3:
            loss = loss + (2 - 1 * torch.mean(L_now))
        elif torch.mean(img_now) < 0.004:
            loss = 0
        else:
            loss = loss + torch.mean(L_now)
    loss = loss / batch
    # print('Light loss:  ', loss)
    return loss


def tensor_gray_YUV(input):
    # return input[:, 0:1, :, :] * 0.299 + input[:, 1:2, :, :] * 0.587 + input[:, 2:3, :, :] * 0.114
    return input[:, 0:1, :, :] * 0.333 + input[:, 1:2, :, :] * 0.333 + input[:, 2:3, :, :] * 0.333


def Light_blocks(L, img, R):
    loss = 0
    batch = img.shape[0]
    avg = nn.AvgPool2d(kernel_size=10)
    for i in range(batch):
        img_now = img[i:i + 1, :, :, :]
        L_now = L[i:i + 1, :, :, :]
        R_now = R[i:i + 1, :, :, :]
        img_now = tensor_gray_YUV(img_now)
        L_now = tensor_gray_YUV(L_now)
        R_now = tensor_gray_YUV(R_now)
        img_avg = avg(img_now)
        L_avg = avg(L_now)

        # R_avg = img_avg / L_avg
        R_avg = avg(R_now)
        I_2 = img_avg * img_avg
        # R_avg = torch.clamp(R_avg, 0.0001, 1)
        zeros = torch.zeros_like(R_avg) + 1.0
        # print('zeros.shape', zeros.shape)
        LI_loss = torch.where(img_avg < 0.5, zeros, L_avg)
        LI_loss = torch.where(L_avg < 1, zeros, L_avg)
        R_Loss = torch.where(R_avg < 0.5, zeros, R_avg)
        # R_Loss = torch.where(R_avg < 0.2, R_avg, R_Loss)
        # R_Loss = torch.where(img_avg < 0.01, zeros, R_Loss)
        # L_Loss = (torch.abs(R_Loss - 0.5) + 0.01) * 0.8
        L_Loss = torch.float_power(LI_loss - 1, 1) * 0.8

        # ranges = 3 * torch.ones_like(L_avg)
        # zeros = torch.zeros_like(L_avg)
        # L_mode1 = torch.where(img_avg > 0.3, L_avg, ranges)
        # L_mode2 = torch.where(img_avg < 0.3, L_avg, zeros)
        # L_mode2 = torch.where(img_avg < 0.004, zeros, L_mode2)
        # L_Loss = torch.float_power((2 - L_mode1), 2) + L_mode2

        loss = loss + torch.mean(L_Loss)
    # loss = loss / batch
    # print('Light loss:  ', loss)
    return loss


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
        loss = Loss_gradient_LIME(R1, k=0.5) + Loss_gradient_LIME(R2, k=0.5) + Loss_gradient_LIME(R3, k=0.5)
        return loss

    def illu_loss(self, x, img):
        img = tensor_gray(img)
        img = img.unsqueeze(1)
        loss = (torch.exp(1.5 * x) - 2) * (torch.exp(1.5 * img) - 2) + 1
        loss = torch.clamp(loss, 0, 5)
        return torch.mean(loss)

    def forward(self, L_list, R_list, img_list, U_list, nR_list, nL_list, imgC_list, img=None):
        ## L_list[0] is MAXc of IMG
        ## img_list[0] is initial input
        Loss_gragent = 0
        Loss_LMSE = 0
        Loss_mean = 0
        Loss_gradient_R = 0
        Loss_img = 0
        Loss_R = 0
        Loss_U = 0
        Loss_RG = 0
        n = len(img_list)
        for i in range(1, n):
            Loss_img = Loss_img + self.L2loss(img_list[i], img_list[i - 1])
            # Loss_mean = Loss_mean + self.mean_loss(L_list[i], img)
        for i in range(0, n - 1):
            # Loss_gradient_R = Loss_gradient_R + self.gloss_R(R_list[i])
            Loss_LMSE = Loss_LMSE + self.L2loss(L_list[i], imgC_list[i])
            Loss_R = Loss_R + self.L2loss(R_list[i], nR_list[i])
            Loss_U = Loss_U + self.illu_loss(U_list[i], img_list[i])
            Loss_RG = Loss_RG + smooth_R(nR_list[i])
            Loss_gragent = Loss_gragent + Loss_gradient_LIME(nL_list[i], k=0.5) + Loss_gradient_LIME(L_list[i], k=0.5)
        return Loss_gragent + Loss_LMSE\
               + Loss_img + (0.5 * Loss_RG + Loss_R) + 0.5 * Loss_U




class LIMEloss_decom(nn.Module):
    def __init__(self):
        super(LIMEloss, self).__init__()
        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()
        self.Avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)


    def forward(self, L_list, R_list, img_list, U_list, nR_list, nL_list, imgC_list, img=None):
        ## L_list[0] is MAXc of IMG
        ## img_list[0] is initial input
        Loss_gragent = 0
        Loss_LMSE = 0
        Loss_img = 0
        n = len(img_list)
        for i in range(1, n):
            Loss_img = Loss_img + self.L2loss(img_list[i], img_list[i - 1])
        for i in range(0, n - 1):
            # Loss_gradient_R = Loss_gradient_R + self.gloss_R(R_list[i])
            Loss_LMSE = Loss_LMSE + self.L2loss(L_list[i], imgC_list[i])
            Loss_gragent = Loss_gragent + Loss_gradient_LIME(nL_list[i], k=0.5) + Loss_gradient_LIME(L_list[i], k=0.5)
        return Loss_gragent + Loss_LMSE


class RES_loss(nn.Module):
    def __init__(self):
        super(RES_loss, self).__init__()
        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()
        self.Lsmooth = SmoothLoss()

    def gloss_R(self, R):
        R1 = R[:, 0:1, :, :]
        R2 = R[:, 1:2, :, :]
        R3 = R[:, 2:3, :, :]
        loss = Loss_gradient_LIME(R1, k=0.5) + Loss_gradient_LIME(R2, k=0.5) + Loss_gradient_LIME(R3, k=0.5)
        return loss

    def forward(self, L, input_list, R_list):
        # Loss_LG = self.gloss(L)
        # Loss_LG = self.Lsmooth(input_list[0], L)
        Loss_LG = self.gloss_R(L)
        Loss_RG = smooth_R(R_list[0])
        Loss_LI = self.L2loss(L, 2 * input_list[0])
        Loss_Light = Light_blocks(L, input_list[0], R_list[0])
        Loss_R = 0
        for i in range(len(input_list)):
            Loss_R = Loss_R + self.L1loss(input_list[i], R_list[i])
        # Loss = Loss_LG + Loss_LI
        Loss = 1 * Loss_LG + 0.5 * Loss_LI + 1 * Loss_Light + 1 * self.L2loss(input_list[0], R_list[0])
        # print('all loss:  ', Loss)
        return Loss


class Denoise_loss(nn.Module):
    def __init__(self):
        super(Denoise_loss, self).__init__()
        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()

    def forward(self, R0, R1):
        Loss_R = self.L2loss(R0, R1)
        Loss_smooth = smooth_R(R1)
        return Loss_R
        # return Loss_R + 0.5 * Loss_smooth


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term


