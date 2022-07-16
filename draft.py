from dataset.dataset import *
from model.SCImodel import *
from model.judgemodel import *
from model.LIMEmodel import *
from util.loss import *
import pytorch_ssim

import cv2
import os
import numpy as np
import torch
import time
import argparse
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from matplotlib import pyplot as plt
from util.loss import *
from pytorch_ssim import SSIM as SSIM


def pix_HSL(r, g, b):
    var_R = r
    var_G = g
    var_B = b
    var_Min = min(var_R, var_G, var_B)  # Min. value of RGB
    var_Max = max(var_R, var_G, var_B)  # Max. value of RGB
    del_Max = var_Max - var_Min  # Delta RGB value
    l = (var_Max + var_Min) / 2.0
    h = 0.0
    s = 0.0
    if del_Max != 0.0:
        if l < 0.5:
            s = del_Max / (var_Max + var_Min)
        else:
            s = del_Max / (2.0 - var_Max - var_Min)
    del_R = (((var_Max - var_R) / 6.0) + (del_Max / 2.0)) / del_Max
    del_G = (((var_Max - var_G) / 6.0) + (del_Max / 2.0)) / del_Max
    del_B = (((var_Max - var_B) / 6.0) + (del_Max / 2.0)) / del_Max
    if var_R == var_Max:
        h = del_B - del_G
    elif var_G == var_Max:
        h = (1.0 / 3.0) + del_R - del_B
    elif var_B == var_Max:
        h = (2.0 / 3.0) + del_G - del_R
    while h < 0.0: h += 1.0
    while h > 1.0: h -= 1.0
    return (h, s, l)


def cal_HSL(input):
    ## calculate HSL from RGB
    R = input[:, 0:1, :, :]
    G = input[:, 1:2, :, :]
    B = input[:, 2:3, :, :]
    Max = torch.max(R, torch.max(G, B))
    Min = torch.min(R, torch.min(G, B))
    Minus = Max - Min
    L = (Max + Min) / 2
    zeros = torch.zeros_like(L)
    S = torch.where(L > 0.5, Minus / (2.001 - 2 * L), Minus / (2 * L + 0.001))
    S = torch.where(L == 0, zeros, S)
    # S = torch.where(Max == Min, zeros, S)
    H = torch.where(Max == R, (G - B) / Minus / 6, zeros)
    H = torch.where(G < B, 1 + (G - B) / Minus / 6, H)
    H = torch.where(Max == G, 1 / 3 + (B - R) / Minus / 6, H)
    H = torch.where(Max == B, 2 / 3 + (R - G) / Minus / 6, H)
    H = torch.where(Max == Min, zeros, H)
    H = H - torch.floor(H)
    return H, S, L


def getparser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default='G:\datasets\LOLdataset\eval15\low', help='location of the data corpus')
    parser.add_argument('--data_path', type=str, default='./1/', help='location of the data corpus')
    parser.add_argument('--save_path', type=str, default='./run', help='location of the data corpus')
    parser.add_argument("--img_size", type=int, default=[300, 400])

    print(parser.parse_args())
    return parser.parse_args()


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def SCITest():
    torch.cuda.empty_cache()
    opt = getparser()

    transforms_ = [
        transforms.Resize(opt.img_size, Image.BICUBIC),
        transforms.ToTensor(),
    ]
    TestDataset = ImageDataset(root=opt.data_path, transform_=transforms_)
    test_imgs = torch.utils.data.DataLoader(
        TestDataset,
        batch_size=1,
        pin_memory=True,
        num_workers=0)

    model = Discriminator()
    model.cuda()

    model.eval()
    with torch.no_grad():
        for i, input in enumerate(test_imgs):
            input = Variable(input['img'], volatile=True).cuda()
            out = model(input)


if __name__ == '__main__':
    opt = getparser()

    # model = LIME_decom(numlayers=3)
    #
    # checkpoint = torch.load('./save/100_LIME_decom.pth')
    # model.load_state_dict(checkpoint['LIME'])

    #
    # figure = plt.figure('1')
    # ax = Axes3D(figure)  # 设置图像为三维格式
    # X = np.arange(0, 2, 0.01)
    # Y = np.arange(0, 1, 0.01)  # X,Y的范围
    # X, Y = np.meshgrid(X, Y)  # 绘制网格
    # # Z = ((np.exp(1.5 * X) - 2) * (np.exp(1.5 * Y) - 2))
    # k = 0.5
    # Z = 1 - 1 / (np.power(X * 0.5 - Y, 2) + 1)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    # figure = plt.figure('2')
    # X = np.arange(0, 1, 0.001)
    # Y = X / (X + 0.1 * np.power(X, 4))
    # plt.plot(X, Y)
    #
    # # 绘制3D图，后面的参数为调节图像的格式
    # plt.show()  # 展示图片


    # torch.cuda.empty_cache()
    # opt = getparser()
    #
    # transforms_ = [
    #     transforms.Resize(opt.img_size, Image.BICUBIC),
    #     transforms.ToTensor(),
    # ]
    # cuda = torch.cuda.is_available()
    # Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # TestDataset = ImageDataset(root=opt.data_path, transform_=transforms_)
    # test_imgs = torch.utils.data.DataLoader(
    #     TestDataset,
    #     batch_size=1,
    #     pin_memory=True,
    #     num_workers=0)
    #
    # print(len(test_imgs))
    # k = 0
    # kmin = 10
    # kmax = 0
    # ks = []
    # avg = nn.AvgPool2d(kernel_size=10)
    # for i, input in enumerate(test_imgs):
    #     input = Variable(input['img'].type(Tensor))
    #     input = input[:, 0:1, :, :] * 0.299 + input[:, 1:2, :, :] * 0.587 + input[:, 2:3, :, :] * 0.114
    #     img_avg = avg(input)
    #     img_avg = torch.mean(img_avg, dim=1)
    #     h = img_avg.shape[1]
    #     w = img_avg.shape[2]
    #     for i in range(h):
    #         for j in range(w):
    #             ks.append(img_avg[0, i, j])
    #     # gray = tensor_gray(input)
    #     k = k + torch.mean(img_avg)
    #     kmin = min(kmin, torch.min(img_avg))
    #     kmax = max(kmax, torch.max(img_avg))
    # k = k / 485
    # print(k)
    # print(kmin)
    # print(kmax)
    # nums = np.zeros(20)
    # print(nums)
    # for k in ks:
    #     nums[min(int(k / 0.05), 19)] = nums[min(int(k / 0.05), 19)] + 1
    # print(nums)

    ### 算术平均得到的亮度（灰度）分布
    # nums_h = [5148, 9533, 16932, 25906, 36508, 44502, 50228, 54749, 52219, 50575,
    #           48183, 45452, 39052, 32955, 25389, 17335, 11594, 7237, 4942, 3561]
    # nums_l = [3.16013e+05, 1.74730e+05, 5.81940e+04, 1.70260e+04, 5.48500e+03, 3.85000e+03, 2.72400e+03, 1.26000e+03, 8.48000e+02, 5.21000e+02,
    #           3.77000e+02, 3.10000e+02, 2.51000e+02, 2.03000e+02, 1.03000e+02, 6.40000e+01, 2.20000e+01, 5.00000e+00, 7.00000e+00, 7.00000e+00]
    ### 加权平均（YUV灰度化）得到的亮度（灰度）分布
    # nums_h = [5223, 9724, 16977, 24800, 34875, 43340, 49890, 54168, 51998, 50208,
    #           48795, 45805, 40008, 34278, 26533, 17734, 11843, 7418, 4985, 3398]
    # nums_l = [3.07856e+05, 1.80886e+05, 6.05660e+04, 1.67460e+04, 5.44600e+03, 4.03500e+03, 2.54700e+03, 1.22200e+03, 7.92000e+02, 5.63000e+02,
    #           4.17000e+02, 3.05000e+02, 2.71000e+02, 2.01000e+02, 6.80000e+01, 5.00000e+01, 1.10000e+01, 5.00000e+00, 8.00000e+00, 5.00000e+00]
    #
    # x = []
    # x0 = 0
    # for i in range(20):
    #     x.append(x0)
    #     x0 += 0.05
    # figure = plt.figure('high')
    # plt.bar(x, nums_h, width=0.01)
    # figure = plt.figure('low')
    # plt.bar(x, nums_l, width=0.01)
    # plt.show()

    #
    # model = Testnet()
    # model.cuda()
    #
    # checkpoint_decom = torch.load('./save/_SCI_model_KD.pth')
    # checkpoint_enhance = torch.load('./save/_SCI_model_EN.pth')
    # # checkpoint_ex = torch.load('./save/100_SCI_model_EX.pth')
    # model.decom.load_state_dict(checkpoint_decom['KD'])
    # model.enhance_net.load_state_dict(checkpoint_enhance['Enhance'])
    # # model.exposure.load_state_dict(checkpoint_ex['Ex'])
    #
    # model.eval()
    #
    # with torch.no_grad():
    #     for i, input in enumerate(test_imgs):
    #         input = Variable(input['img'], volatile=True).cuda()
    #
    #         R, L = model.decom(input)
    #         U = model.enhance_net(R)
    #         print(U.shape)
    #         U = U[0, :, :, :]
    #         U = U[0, :, :]
    #         print(U)
    #         U = U.cpu().detach().numpy()
    #         np.savetxt('np_U.txt', U, fmt='%f')
    #         L = L[0, :, :, :]
    #         L = L[0, :, :]
    #         print(L)
    #         L = L.cpu().detach().numpy()
    #         np.savetxt('np_L.txt', L, fmt='%f')

    a = torch.randn([3, 5, 6, 7])
    b, c, w, h = a.shape
    a = a.view(b, c, w*h)
    print(a.shape)

    # x = torch.randn([8, 1, 20, 30], dtype=torch.float32).cuda()
    # gx = gradient(x, 'x')
    # gy = gradient(x, 'y')
    # print(gx.shape)
    # print(gy.shape)

    # x = torch.randn([1, 32, 16, 16]).cuda()
    # avg = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    # max = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    # min = -F.max_pool2d( -x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    # print('min:  ', min.shape)
    # # conv.weight.data = kernel
    # gx = gradient(x, 'x')
    # # gx = conv(x)
    # gra = gradient(x, 'x') * torch.exp(-10 * gradient(x, 'x')) + gradient(x, 'y') * torch.exp(-10 * gradient(x, 'y'))
    # gra = F.avg_pool2d( gra, (gra.size(2), gra.size(3)), stride=(gra.size(2), gra.size(3)))
    # print('avg:  ', avg.shape)
    # print('max:  ', max.shape)
    # print('gra:  ', gra.shape)
    # all = torch.cat([avg, max, gra, min], 2)
    # all = all.squeeze(3)
    # all = torch.randn([32, 256, 3]).cuda()
    # print('all:  ', all.shape)
    # lin = nn.Linear(3, 1).cuda()
    # all = lin(all)
    # print('all:  ', all.shape)


    #
    # torch.cuda.empty_cache()
    # MSE = nn.MSELoss()
    # transforms_ = [
    #     # transforms.Resize(opt.img_size, Image.BICUBIC),
    #     transforms.ToTensor(),
    # ]
    # cuda = torch.cuda.is_available()
    # Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # TestDataset = ImageDataset(root=opt.data_path, transform_=transforms_)
    # test_imgs = torch.utils.data.DataLoader(
    #     TestDataset,
    #     batch_size=1,
    #     pin_memory=True,
    #     num_workers=0)
    # print(len(test_imgs))
    # model = RES_decom()
    # # denoise_model = D_net()
    # model.cuda()
    # # denoise_model.cuda()
    # checkpoint = torch.load('./save/200_RES_decom.pth')
    # model.load_state_dict(checkpoint['RES'])
    # # checkpoint1 = torch.load('./save/100_Denoise.pth')
    # # denoise_model.load_state_dict(checkpoint1['Denoise'])
    # model.eval()
    # # denoise_model.eval()
    # now = 0
    # ssim = 0
    # SSIM = SSIM()
    # with torch.no_grad():
    #     for i, input in enumerate(test_imgs):
    #         time0 = time.time()
    #         input = Variable(input['img'].type(Tensor))
    #         for i in range(1):
    #             R, L = model(input)
    #             # HR, _, _ = cal_HSL(R)
    #             # Hinput, _, _ = cal_HSL(input)
    #             # print(MSE(HR, Hinput))
    #             # break
    #             # R = denoise_model(R)
    #             # input = R
    #             # nowssim = SSIM(R, input)
    #             # print(nowssim.cpu().detach().numpy())
    #             # ssim = ssim + nowssim
    #             sample_single_img(now, R[0, :, :, :], 'R', './run/test')
    #             sample_single_img(now, input[0, :, :, :], '', './run/test')
    #             sample_single_img(now, L[0, :, :, :], 'L', './run/test')
    #             now = now + 1
    #         timenow = time.time() - time0
    #         print(timenow)
    # # print(ssim.cpu().detach().numpy() / now)
    # # #
    # #








