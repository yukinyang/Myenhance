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
    parser.add_argument('--data_path', type=str, default='./testimg', help='location of the data corpus')
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
    # img_list = sorted(glob.glob('G:/datasets/LOLdataset/our485/low' + "/*.*"))
    # print(img_list)
    # for img in img_list:
    #     name = img
    #     newname = img[0 : len(name) - 4] + '_l.png'
    #     # print(newname)
    #     os.rename(img, newname)


    # dirs = os.listdir('./run//')
    # print(dirs)
    # i = 0
    # name = 'runs'
    # for i in range(1, 10000):
    #     newdir = name + str(i)
    #     if newdir not in dirs:
    #         name = newdir
    #         break
    # print(name)


    # img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
    # # cv2.imshow('img', img)
    # # cv2.waitKey()
    # np_img = np.array(img)
    # np.savetxt('np_img.txt', np_img, fmt='%d')

    figure = plt.figure('1')
    ax = Axes3D(figure)  # 设置图像为三维格式
    X = np.arange(0, 1, 0.01)
    Y = np.arange(0, 1, 0.01)  # X,Y的范围
    X, Y = np.meshgrid(X, Y)  # 绘制网格
    Z = ((np.exp(1.5 * X) - 2) * (np.exp(1.5 * Y) - 2)) # f(x,y)=(sin(x)*sin(y))/(x*y),注意括号
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    figure = plt.figure('2')
    X = np.arange(0, 1, 0.01)
    Y = (np.exp(1.5 * X) - 2)  # f(x,y)=(sin(x)*sin(y))/(x*y),注意括号
    plt.plot(X, Y)

    # 绘制3D图，后面的参数为调节图像的格式
    plt.show()  # 展示图片




