'''
Written by Ding Yuhui
'''
import torch.nn as nn
import torch
from module import ScaleTransfer, Subnet, StemBlock
from data import VOC
from torchvision.models import densenet169
from util import Anchor
from module import Detect
import os

class STDN(nn.Module):
    def __init__(self, cfg=VOC):
        super().__init__()
        anchor = Anchor()
        self.image_size = cfg['image_size']
        self.anchors, anchors_per_map = anchor.generate()
        densenet = densenet169(True).features
        seq = [StemBlock()]
        for i, (key, m) in enumerate(densenet._modules.items()):
            if i < 4:
                continue
            if key == 'denseblock4':
                break
            else:
                seq.append(m)
        self.dense_forward = nn.Sequential(*seq)
        self.last_denseblock = densenet.denseblock4
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(cfg['num_classes'], 0, 200, 0.01, 0.45)
        scale_modules = [
            nn.AvgPool2d,
            nn.AvgPool2d,
            nn.AvgPool2d,
            nn.Sequential,
            ScaleTransfer,
            ScaleTransfer
        ]

        for i, (m, p) in enumerate(zip(scale_modules, cfg['scale_modules'])):
            self.add_module('scale' + str(i), m(**p))

        for i, (c, n, f) in enumerate(zip(cfg['subnet_modules']['in_channels'], reversed(anchors_per_map), reversed(cfg['feature_maps']))):
            self.add_module('loc' + str(i), Subnet(c, n * 4 // f**2))
            self.add_module('clf' + str(i), Subnet(c, n * cfg['num_classes'] // f**2))

    def forward(self, input):
        temp = self.dense_forward(input)
        output_loc = []
        output_clf = []
        i = 0
        for key, m in self.last_denseblock._modules.items():
            n = int(key[10:])
            temp = m(temp)
            if (n % 5 == 0 and n != 30) or n == 32:
                sm = self._modules.get('scale' + str(i))
                temp2 = sm(temp)
                temp3 = self._modules.get('loc' + str(i))(temp2)
                output_loc.append(temp3.permute(0, 2, 3, 1).contiguous().view(temp3.size(0), -1, 4))
                temp3 = self._modules.get('clf' + str(i))(temp2)
                output_clf.append(temp3.permute(0, 2, 3, 1).contiguous().view(temp3.size(0), -1, 21))
                i += 1
        if self.training:
            return torch.cat(tuple(reversed(output_loc)), 1), torch.cat(tuple(reversed(output_clf)), 1)
        return self.detect(torch.cat(tuple(reversed(output_loc)), 1),
                           self.softmax(torch.cat(tuple(reversed(output_clf)), 1)), self.anchors)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


if __name__ == '__main__':
    anchor = Anchor()
    boxes, num = anchor.generate()
    stdn = STDN(num)
    out1, out2 = stdn(torch.rand(2, 3, 300, 300))
    print(out1.size())
    print(out2.size())

