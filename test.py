from dataset.dataset import *
from model.SCImodel import *

import cv2
import numpy as np
import torch
import argparse
import tensorflow as tf
from torch.autograd import Variable
from matplotlib import pyplot as plt
from util.loss import *


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./testimg', help='location of the data corpus')
    parser.add_argument('--save_path', type=str, default='./run', help='location of the data corpus')

    print(parser.parse_args())
    return parser.parse_args()


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


if __name__ == '__main__':
    opt = getparser()
    TestDataset = TestLoader(img_dir=opt.data_path, task='test')

    test_imgs = torch.utils.data.DataLoader(
        TestDataset,
        batch_size=1,
        pin_memory=True,
        num_workers=0)

    model = Testnet()
    model.cuda()

    checkpoint_decom = torch.load('./save/200_SCI_model_KD.pth')
    checkpoint_enhance = torch.load('./save/200_SCI_model_EN.pth')
    model.decom.load_state_dict(checkpoint_decom['KD'])
    model.enhance_net.load_state_dict(checkpoint_enhance['Enhance'])

    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_imgs):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('\\')[-1].split('.')[0]
            out = model(input, 1)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = opt.save_path + '/' + u_name
            u_path_i = opt.save_path + '/' + 'i_' + u_name
            save_images(out, u_path)






