import torch.nn as nn
import torch.nn.functional as F
import torch


class Decom(nn.Module):
    def __init__(self, layers):
        super(Decom, self).__init__()

        self.layers = layers
        self.inchannels = 3
        self.outchannels = 7

        # self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.convs = self.make_convs(self.inchannels, self.outchannels, self.layers)
        self.test_conv = nn.Conv2d(self.inchannels, self.outchannels, kernel_size=3, stride=1, padding=1)

    def make_convs(self, inchannels, outchannels, numlayers):
        layers = []
        midchannels = 64
        layers.append(nn.Sequential(
            nn.Conv2d(inchannels, midchannels, kernel_size=7, padding=3, stride=1)
        ))
        for i in range(0, numlayers):
            layers.append(nn.Sequential(
                nn.Conv2d(midchannels, midchannels, kernel_size=3, padding=1, stride=1)
            ))
        layers.append(nn.Sequential(
            nn.Conv2d(midchannels, outchannels, kernel_size=3, padding=1, stride=1)
        ))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = input
        # x = self.convs(x)
        x = self.test_conv(x)
        x = self.sigmoid(x)
        # R = x[:, 0:3, :, :]
        # L = x[:, 3:4, :, :]
        # E = x[:, 4:7, :, :] - 0.5
        # return R, L, E
        return x



class enhance(nn.Module):
    def __init__(self):
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input):
        x = input
        return x