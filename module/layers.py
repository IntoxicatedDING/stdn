'''
Written by Ding Yuhui
'''
import torch.nn as nn
import numpy as np
import itertools
from torch.autograd import Variable
import torch


class ScaleTransfer(nn.Module):
    def __init__(self, ratio, channels, size):
        super().__init__()
        self.__ratio = ratio
        self.__card = [channels, size * ratio, size * ratio]
        self.__idx_c, self.__idx_y, self.__idx_x = ScaleTransfer.__cartesian_product(self.__card)

    def forward(self, input):
        # r = self.__ratio
        # C = self.__card[0]
        # size = self.__card[1]
        # subpixel = torch.cat([torch.reshape(a, (-1, size, r, C)) for a in torch.chunk(input.permute(0, 2, 3, 1), size // r, dim=2)], dim=2).permute(0, 3, 1, 2)
        # return subpixel
        # variable = input.view([input.size(0)] + self.__card)
        variable = Variable(torch.zeros([input.size(0)] + self.__card)).type(input.type())
        # channels = self.__card[0]
        idx_c, idx_y, idx_x, r = self.__idx_c, self.__idx_y, self.__idx_x, self.__ratio
        variable[:, idx_c, idx_y, idx_x] = \
            input[:, r * (idx_x % r) + (idx_y % r) + idx_c * r**2, idx_y // r, idx_x // r]
        return variable

    @staticmethod
    def __cartesian_product(card):
        product = itertools.product(*[np.arange(c, dtype=np.int32) for c in card])
        ret = []
        for p in product:
            ret.append(np.array(p))
        ret = np.stack(ret, axis=0)
        return [ret[:, i] for i in range(ret.shape[1])]


class Subnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, input):
        return self.net(input)


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, input):
        return self.net(input)



if __name__ == '__main__':
    pass