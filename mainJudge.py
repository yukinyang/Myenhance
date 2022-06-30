from model.model import *
from util.util import *
from util.loss import *
from dataset.dataset import *
from model.SCImodel import *
from test import *
from model.judgemodel import *

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_cpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default='imgs/')
    parser.add_argument("--img_size", type=int, default=[300, 400])
    parser.add_argument("--decay_epoch", type=int, default=200)

    print(parser.parse_args())
    return parser.parse_args()


def Train_Discriminator():
    opt = getparser()
    model = Discriminator()
    model.cuda()














