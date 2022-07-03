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

    def forword(self, L_list):
        # L_list[0] is MAXc of IMG
        Loss_gragent = 0
        Loss_LMSE = 0
        n = len(L_list)
        for i in range(1, n):
            Loss_LMSE = Loss_LMSE + self.L2loss(L_list[i], L_list[i - 1])
            Loss_gragent = Loss_gragent + Loss_gradient_LIME(L_list[i], k=0.5)
        return Loss_gragent + Loss_LMSE










