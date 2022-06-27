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
            nn.Conv2d(inchannels, midchannels, kernel_size=7, padding=3, stride=1),
            # nn.InstanceNorm2d(midchannels),
        ))
        for i in range(0, numlayers):
            layers.append(nn.Sequential(
                nn.Conv2d(midchannels, midchannels, kernel_size=3, padding=1, stride=1),
                # nn.InstanceNorm2d(midchannels),
            ))
        layers.append(nn.Sequential(
            nn.Conv2d(midchannels, outchannels, kernel_size=3, padding=1, stride=1),
            # nn.InstanceNorm2d(outchannels),
        ))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = input
        x = self.convs(x)
        # x = self.test_conv(x)
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



class KD_decom(nn.Module):
    def __init__(self):
        super(KD_decom, self).__init__()
        self.inch = 3
        self.outch1 = 3
        self.outch2 = 1
        self.outch3 = 3

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(self.inch, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # R channel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(32 + 32, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv6 = nn.Conv2d(32, 3, kernel_size=1, stride=1)
        # L channel
        self.l_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # self.l_conv2 = nn.Conv2d(32 + 32, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.l_conv2 = nn.Conv2d(32 + 32, 1, kernel_size=1, stride=1)
        # E channel
        self.e_convs = self.make_convs(32, 3, 5)

    def make_convs(self, inch, outch, num):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(inch, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        ))
        for i in range(0, num - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ))
        layers.append(nn.Sequential(
            nn.Conv2d(32, outch, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        ))
        return nn.Sequential(*layers)


    def forward(self, input):
        x = input

        mid0 = self.conv1(x)
        mid0 = self.relu(mid0)
        # print("mid0:   ", mid0.shape)

        mid1 = self.pool1(mid0)
        mid1 = self.conv2(mid1)
        mid1 = self.relu(mid1)
        # print("mid1:   ", mid1.shape)

        mid2 = self.pool2(mid1)
        mid2 = self.conv3(mid2)
        # print("mid2:   ", mid2.shape)

        mid3 = self.up1(mid2)
        mid3 = torch.cat([mid1, mid3], 1)
        mid3 = self.conv4(mid3)
        mid3 = self.relu(mid3)
        # print("mid3:   ", mid3.shape)

        mid4 = self.up2(mid3)
        mid4 = torch.cat([mid0, mid4], 1)
        mid4 = self.conv5(mid4)
        mid4 = self.relu(mid4)
        # print("mid4:   ", mid4.shape)

        R = self.conv6(mid4)
        R = self.sigmoid(R)
        # print("R:   ", R.shape)

        mid_l1 = self.l_conv1(mid0)
        mid_l1 = self.relu(mid_l1)
        # print("midl1:   ", mid_l1.shape)

        L = torch.cat([mid4, mid_l1], 1)
        L = self.l_conv2(L)
        L = self.sigmoid(L)

        E = self.e_convs(mid0)
        E = self.sigmoid(E) - 0.5

        return R, L, E



