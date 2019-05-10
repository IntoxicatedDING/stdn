'''
Adapted from https://github.com/amdegroot/ssd.pytorch
'''
from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
from data import VOC


class Anchor:

    def __init__(self, cfg=VOC):
        self.image_size = cfg['image_size']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['feature_maps'])
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratio = cfg['aspect_ratio']

    def generate(self):
        anchors = []
        num = []
        for k, f in enumerate(self.feature_maps):
            cnt = 0
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                anchors += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                anchors += [cx, cy, s_k_prime, s_k_prime]
                cnt += 2
                # rest of aspect ratios
                for ar in self.aspect_ratio:
                    anchors += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    anchors += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                    cnt += 2
            num.append(cnt)
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output, num


if __name__ == '__main__':
    import numpy as np
    anchor = Anchor()
    output, num = anchor.generate()
    print(num)
    print(output.size())
    torch.masked_select()