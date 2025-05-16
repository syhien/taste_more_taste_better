# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

# from ..builder import HEADS

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


# @HEADS.register_module()
class CountingHead(nn.Module):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self, in_channels, out_channels, inter_layer=[64, 32, 16]):
        super(CountingHead, self).__init__()

        self.out_channels = out_channels
        self.counter_inchannels = in_channels
        self.inter_layer = inter_layer

        if self.out_channels <= 0:
            raise ValueError(
                f"num_classes={self.out_channels} must be a positive integer"
            )
        self.count = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                self.counter_inchannels,
                self.inter_layer[0],
                3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(self.inter_layer[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.inter_layer[0],
                self.inter_layer[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(self.inter_layer[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                self.inter_layer[1],
                self.inter_layer[2],
                3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(self.inter_layer[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.inter_layer[2],
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, y):
        return self.count(y)
