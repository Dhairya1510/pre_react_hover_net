import math
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet

from .net_utils import DenseBlock, UpSample2x, TFSamepaddingLayer, ResidualBlock
from .utils import crop_op, crop_to_shape


class RES(nn.Module):
    def __init__(self,freeze=False,training=True):
        super(RES, self).__init__()
        self.freeze = freeze
        self.training=training
        #self.input_ch= input_ch
        module_list = [
            ("/", nn.Conv2d(3, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        self.pad = TFSamepaddingLayer(ksize=7,stride=1)
        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

    def forward(self, x):
        if self.training:
            d0 = self.pad(x)
            d0 = self.conv0(d0)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
        else:
            d0 = self.pad(x)
            d0 = self.conv0(d0)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
        return d0, d1, d2, d3

def backbone(pretrained_path,freeze=False,training=True):
    if pretrained_path is not None:
       model = RES(freeze, training)
       state = torch.load(pretrained_path)['desc']
       for k,v in state.copy().items():
          if k=='upsample2x.unpool_mat':
            state_dict = state.pop('upsample2x.unpool_mat')
       model.load_state_dict(state)
    else:
        model=RES(freeze,training)
    return model

####
class HoVerNetExt(nn.Module):
    """Initialise HoVer-Net."""

    def __init__(
            self,
            num_types=None,
            freeze=False,
            pretrained_backbone=None,
            training=True,
            ):
        super().__init__()
        self.freeze = freeze 
        self.num_types = num_types
        self.training = training 
        self.output_ch = 3 if num_types is None else 4  

        self.backbone = backbone(pretrained_path=pretrained_backbone,freeze=freeze,training=training)

        self.conv_bot = nn.Conv2d(
            2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            pad = ksize // 2
            module_list = [
                nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(256, [1, ksize], [128, 32], 8, split=4),
                nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            ]
            u3 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(128, [1, ksize], [128, 32], 4, split=4),
                nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            ]
            u2 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
            ]
            u1 = nn.Sequential(*module_list)

            module_list = [
                nn.BatchNorm2d(64, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
            ]
            u0 = nn.Sequential(*module_list)

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
            )
            return decoder

        ksize = 3
        if num_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=num_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()

    def forward(self, imgs):
        imgs = imgs / 255.0  # to 0-1 range to match XY

        d0, d1, d2, d3 = self.backbone(imgs)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            #u0 = self.upsample2x(u1) ##Not added with anything can be added with conv0 output
            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, **kwargs):
    return HoVerNetExt(**kwargs)
