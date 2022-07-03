from dataset.dataset import *
from model.SCImodel import *
from model.judgemodel import *
from util.loss import *
import pytorch_ssim

import cv2
import os
import numpy as np
import torch
import argparse
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from matplotlib import pyplot as plt
from util.loss import *


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./1', help='location of the data corpus')
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

    # figure = plt.figure('1')
    # ax = Axes3D(figure)  # 设置图像为三维格式
    # X = np.arange(0, 1, 0.01)
    # Y = np.arange(0, 1, 0.01)  # X,Y的范围
    # X, Y = np.meshgrid(X, Y)  # 绘制网格
    # Z = ((np.exp(1.5 * X) - 2) * (np.exp(1.5 * Y) - 2)) # f(x,y)=(sin(x)*sin(y))/(x*y),注意括号
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    #
    # figure = plt.figure('2')
    # X = np.arange(0, 1, 0.01)
    # Y = (np.exp(1.5 * X) - 2)  # f(x,y)=(sin(x)*sin(y))/(x*y),注意括号
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
    # TestDataset = ImageDataset(root=opt.data_path, transform_=transforms_)
    # test_imgs = torch.utils.data.DataLoader(
    #     TestDataset,
    #     batch_size=1,
    #     pin_memory=True,
    #     num_workers=0)
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


    img = torch.randn([1, 20, 30])
    sample_gray_img(1, img, 'gray', './run')














