import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, Multi_Concat_Block, Conv, SiLU, Transition_Block, autopad


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.decoder(x)


class YoloUnet(nn.Module):
    def __init__(self):
        super(YoloUnet, self).__init__()
        self.backbone = Backbone(32, 32, 4, phi='l')
        self.up1 = DecoderBlock(1024, 512)  # 输入大小20x20，输出大小40x40
        self.up2 = DecoderBlock(512, 256)  # 输入大小40x40，输出大小80x80
        self.up3 = DecoderBlock(256, 128)  # 输入大小80x80，输出大小160x160
        self.up4 = DecoderBlock(128, 64)  # 输入大小160x160，输出大小320x320
        self.up5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  # 上采样至640x640，通道数为3

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        # 80	80	512
        # 40	40	1024
        # 20    20	1024

        x = feat3;
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        out = x
        return out
