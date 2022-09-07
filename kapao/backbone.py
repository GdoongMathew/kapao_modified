import math
from typing import Tuple
from typing import List

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models import mobilenet_v3_small

from .conv_module import \
    BottleneckC3SP, \
    ConvBnAct, \
    SPP, \
    SPPF, \
    Focus


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def mobilenet_small(
        arch='small',
        pretrained=True,
        layers=(-10, -5, -1)
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    backbone = mobilenet_v3_small(pretrained=pretrained)
    p4 = backbone.features[layers[0]]  # 24, 96, 96
    p5 = backbone.features[layers[1]]  # 48, 48, 48
    p6 = backbone.featires[layers[2]]  # 576, 24, 24

    return p4, p5, p6


class YoloV56(nn.Module):
    """
    YoloV5_v6 implementation.
    Output tensors are sorted in ascending order by its number of channels.

    """
    def __init__(
            self,
            module_ch: List[int] = [64, 128, 256, 512, 768, 1024],
            repeat_n: List[int] = [3, 9, 9, 3, 3],
    ):
        super(YoloV56, self).__init__()

        self.stem = ConvBnAct(
            c_in=3,
            c_out=module_ch[0],
            kernel_size=6,
            padding=2,
            stride=2
        )
        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBnAct(c_in=c_in, c_out=c_out, kernel_size=3, padding=1, stride=2),
                    BottleneckC3SP(c_in=c_out, c_out=c_out, n=n, expansion=.5, shortcut=True),
                )
                for c_in, c_out, n in zip(module_ch[:-1], module_ch[1:], repeat_n)
            ]
        )

        self.sppf = SPPF(
            c_in=module_ch[-1], c_out=module_ch[-1], pool_size=5
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        ret = []
        for module in self.module_list:
            x = module(x)
            ret.append(x)

        ret[-1] = self.sppf(ret[-1])

        return ret

    #     self.module_list = nn.ModuleList(
    #         [
    #             nn.Sequential(
    #                 ConvBnAct(c_in=c_in, c_out=c_out, kernel_size=3, padding=1, stride=2),
    #                 BottleneckC3SP(c_in=c_out, c_out=c_out, n=n, expansion=.5, shortcut=True),
    #             )
    #             for c_in, c_out, n in zip(module_ch[:-2], module_ch[1:-1], repeat_n[:-1])
    #         ]
    #     )
    #
    #     self.p6 = nn.Sequential(
    #         ConvBnAct(c_in=module_ch[-2], c_out=module_ch[-1], kernel_size=3, stride=2),
    #         SPPF(c_in=module_ch[-1], c_out=module_ch[-1], pool_size=5),
    #         BottleneckC3SP(c_in=module_ch[-1], c_out=module_ch[-1], n=repeat_n[-1], shortcut=False),
    #     )
    #
    # def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
    #     x = self.stem(x)
    #     ret = []
    #     for module in self.module_list:
    #         x = module(x)
    #         ret.append(x)
    #
    #     ret.append(self.p6(ret[-1]))
    #
    #     return ret
