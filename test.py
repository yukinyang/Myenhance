from dataset.dataset import *
from model.SCImodel import *

import cv2
import numpy as np
import torch
import argparse
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


def DecomTest(model, test_dir, save_dir, load_dir):
    torch.cuda.empty_cache()
    opt = getparser()
    transforms_ = [
        transforms.Resize(opt.img_size, Image.BICUBIC),
        transforms.ToTensor(),
    ]
    TestDataset = ImageDataset(root=test_dir, transform_=transforms_)
    test_imgs = torch.utils.data.DataLoader(
        TestDataset,
        batch_size=1,
        pin_memory=True,
        num_workers=0)

    checkpoint = torch.load(load_dir)
    model.load_state_dict(checkpoint['KD'])

    model.eval()
    now = 1
    with torch.no_grad():
        for i, input in enumerate(test_imgs):
            input = Variable(input['img'], volatile=True).cuda()
            R, L = model(input)
            if len(L.shape) == 4:
                R.squeeze(0)
                L.squeeze(0)
            L = torch.cat([L, L, L], 1)
            u_name = '%s.png' % (str(now) + '_gen')
            u_name_R = '%s_R.png' % (str(now) + '_R')
            u_path = save_dir + '/' + u_name
            u_path_R = save_dir + '/' + u_name_R
            save_images(R * L, u_path)
            save_images(R, u_path_R)
            now = now + 1


def SCITest(Savedir):
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

    model = Testnet()
    model.cuda()

    checkpoint_decom = torch.load('./save/100_SCI_model_KD_yuan.pth')
    checkpoint_enhance = torch.load('./save/100_SCI_model_EN_yuan.pth')
    checkpoint_ex = torch.load('./save/100_SCI_model_EX.pth')
    model.decom.load_state_dict(checkpoint_decom['KD'])
    model.enhance_net.load_state_dict(checkpoint_enhance['Enhance'])
    model.exposure.load_state_dict(checkpoint_ex['Ex'])

    model.eval()
    for rr in range(0, 51):
        if rr % 5 != 0:
            continue
        now = 0
        with torch.no_grad():
            for i, input in enumerate(test_imgs):
                input = Variable(input['img'], volatile=True).cuda()
                image_name = str(now)
                out, R = model(input, rr)
                u_name = '%s.png' % (image_name + '_' + str(rr))
                u_name_R = '%s_R.png' % (image_name)
        #         print('processing {}'.format(u_name))
                u_path = Savedir + '/' + u_name
                u_path_R = Savedir + '/' + u_name_R
        #         u_path_i = opt.save_path + '/' + 'i_' + u_name
                save_images(out, u_path)
                save_images(R, u_path_R)
                now = now + 1


if __name__ == '__main__':
    run_dir = get_dir_name('./run', 'SCItest')
    os.makedirs(run_dir)
    SCITest(run_dir)



