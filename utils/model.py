# modify from https://github.com/facebookresearch/moco/blob/main/moco/builder.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lenet import LeNet
from .resnet import *
from .wideResnet import WideResNet
from .wideResnetVar import WideResNetVar


class Model(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        
        # encoder
        self.encoder_q = base_encoder(name=args.arch, num_class=args.num_class)

    def forward(self, img_q):
        q_f = self.encoder_q(img_q)
        return q_f


class sspll(nn.Module):
    def __init__(self, name='resnet18', num_class=10):
        super(sspll, self).__init__()

        if name in ['resnet18', 'resnet34']:
            model_fun, dim_in = model_dict[name]
            self.encoder = model_fun()
            self.classifier = nn.Linear(dim_in, num_class)
        elif name == 'WRN_28_2':
            self.encoder = WideResNet(depth=28, num_classes=num_class, widen_factor=2)
            self.classifier = Identity()
        elif name == 'WRN_28_8':
            self.encoder = WideResNet(depth=28, num_classes=num_class, widen_factor=8)
            self.classifier = Identity()
        elif name == 'WRN_37_2':
            self.encoder = WideResNetVar(first_stride=2, num_classes=num_class, depth=28, widen_factor=2)
            self.classifier = Identity()

        elif name == 'lenet':
            self.encoder = LeNet(out_dim=num_class, in_channel=1, img_sz=28)
            self.classifier = nn.Linear(500, num_class)

    def forward(self, x):
        feat = self.encoder(x)
        f = self.classifier(feat)
        return f
