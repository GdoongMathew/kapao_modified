from typing import \
    Union, \
    List, \
    Tuple, \
    OrderedDict

import math

import torch
from torch import nn
from torch.nn import functional as F


class Focus(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            kernel_size: Union[int, List] = 1,
            **kwargs
    ):
        super(Focus, self).__init__()
        self.conv = ConvBnAct(c_in * 4, c_out, kernel_size=kernel_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(
            torch.cat([
                x[..., ::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, ::2],
                x[..., 1::2, 1::2]
            ], dim=1)
        )


class ConvBnAct(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 kernel_size: Union[int, List] = 3,
                 stride: int = 1,
                 padding: Union[None, int] = None,
                 groups: int = 1,
                 act=True
                 ):
        super(ConvBnAct, self).__init__()

        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
        self.conv = nn.Conv2d(c_in,
                              c_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU() if act else (act if isinstance(act, nn.Module) else None)

    def _forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.act is not None else x

    def forward(self, x):
        return self._forward(x) if self.training else self.fused_forward(x)

    def fused_forward(self, x):
        x = self.conv(x)
        return self.act(x) if self.act is not None else x


class DepthWiseConvBnAct(ConvBnAct):
    """Depth-wise Convolution BatchNorm Activation"""

    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size: Union[int, List] = 3,
                 stride: int = 1,
                 padding: Union[None, int] = None,
                 act=True,
                 ):
        super(DepthWiseConvBnAct, self).__init__(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            groups=math.gcd(c_in, c_out),
            padding=padding,
            act=act
        )


class PointWiseConvBnAct(ConvBnAct):
    """Point-wise Convolution BatchNorm Activation"""

    def __init__(self,
                 c_in,
                 c_out,
                 act=True,
                 ):
        super(PointWiseConvBnAct, self).__init__(
            c_in,
            c_out,
            kernel_size=1,
            padding=None,
            act=act
        )


class Bottleneck(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            shortcut: bool = True,
            groups: int = 1,
            exp: float = .5
    ):
        super(Bottleneck, self).__init__()
        _c = int(exp * c_out)
        self.conv_1 = ConvBnAct(c_in, _c, kernel_size=1, groups=1)
        self.conv_2 = ConvBnAct(_c, c_out, kernel_size=3, groups=groups)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        _x = x
        _x = self.conv_2(self.conv_1(x))
        _x = _x + x if self.add else x
        return _x


class BottleneckCSP(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            n: int = 1,
            shortcut: bool = True,
            inplace: bool = True,
            groups=1,
            exp=.5
    ):
        super(BottleneckCSP, self).__init__()
        _c = int(c_out * exp)
        self.conv_1 = nn.Conv2d(c_in, _c, kernel_size=1)
        self.skip_conv = nn.Conv2d(c_in, _c, kernel_size=1, bias=False)
        self.bottle_conv = nn.Conv2d(_c, _c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(_c * 2)
        self.act = nn.LeakyReLU(negative_slope=.1, inplace=inplace)
        self.out_conv = ConvBnAct(_c * 2, _c, kernel_size=1)

        self.bottlenecks = nn.Sequential(*[Bottleneck(_c,
                                                      _c,
                                                      shortcut=shortcut,
                                                      groups=groups,
                                                      exp=exp) for _ in range(n)])

    def forward(self, x):
        _x = self.skip_conv(x)
        x = self.bottle_conv(self.bottlenecks(self.conv_1(x)))
        x = self.out_conv(self.act(self.bn(torch.concat([x, _x], dim=1))))
        return x


class BottleneckC3SP(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            n: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            expansion: float = .5
    ):
        super(BottleneckC3SP, self).__init__()
        c_ = int(c_out * expansion)
        self.conv1 = ConvBnAct(c_in, c_, kernel_size=1, stride=1)
        self.conv2 = ConvBnAct(c_in, c_, kernel_size=1, stride=1)
        self.conv3 = ConvBnAct(c_ * 2, c_out, kernel_size=1, stride=1)

        self.bottleneck_list = nn.Sequential(*[
            Bottleneck(c_, c_, shortcut=shortcut, groups=groups, exp=expansion) for _ in range(n)
        ])

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bottleneck_list(x1)
        x2 = self.conv2(x)
        return self.conv3(torch.cat([x1, x2], dim=1))


class SpatialPyramidPooling(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            pool_size: Tuple[int, int, int] = (5, 9, 13)
    ):
        super(SpatialPyramidPooling, self).__init__()
        _c = c_in // 2
        self.conv_in = ConvBnAct(
            c_in,
            _c,
            kernel_size=1
        )
        self.conv_out = ConvBnAct(
            _c * (len(pool_size) + 1),
            c_out,
            kernel_size=1
        )
        self.pooling = nn.ModuleList([
            nn.MaxPool2d(kernel_size=p, padding=p // 2)
            for p in pool_size
        ])

    def forward(self, x):
        x = self.conv_in(x)
        pool_x = [pool(x) for pool in self.pooling]
        x = torch.concat(pool_x + [x], dim=1)
        return self.conv_out(x)


SPP = SpatialPyramidPooling


class SpatialPyramidPoolingFast(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            pool_size: int = 5
    ):
        super(SpatialPyramidPoolingFast, self).__init__()
        _c = c_in // 2
        self.conv_in = ConvBnAct(
            c_in,
            _c,
            kernel_size=1
        )
        self.conv_out = ConvBnAct(
            _c * 4,
            c_out,
            kernel_size=1
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=pool_size,
            padding=pool_size // 2,
            stride=1
        )

    def forward(self, x):
        x = self.conv_in(x)
        x1 = self.pooling(x)
        x2 = self.pooling(x1)
        x3 = self.pooling(x2)
        return self.conv_out(torch.cat([x, x1, x2, x3], dim=1))


SPPF = SpatialPyramidPoolingFast


class PSAChannel(nn.Module):
    """ Polarized Self-Attention Channel only module from https://arxiv.org/abs/2107.00782
    """
    def __init__(
            self,
            in_channels: int,
            channel_factor: float = .5
    ):
        super(PSAChannel, self).__init__()
        self._ch = int(in_channels * channel_factor)
        self.wv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self._ch,
            kernel_size=1,
            stride=1
        )

        self.wq = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1
        )

        self.wz = nn.Conv2d(
            in_channels=self._ch,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )

        self.norm = nn.LayerNorm(normalized_shape=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        xv = torch.reshape(self.wv(x), (bs, self._ch, -1))
        xq = torch.reshape(self.wq(x), (bs, -1, 1)).softmax(dim=1)
        x_ = torch.matmul(xv, xq)
        x_ = self.norm(self.wz(x_[..., None])).sigmoid()
        return x * x_


class PSASpatial(nn.Module):
    """ Polarized Self-Attention Spatial only module from https://arxiv.org/abs/2107.00782
    """
    def __init__(
            self,
            in_channels: int,
            channel_factor: float = .5
    ):
        super(PSASpatial, self).__init__()
        self._ch = int(in_channels * channel_factor)
        self.wq = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self._ch,
            kernel_size=1,
        )

        self.wv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self._ch,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, _, *dim = x.shape
        xq = torch.reshape(
            F.max_pool2d(self.wq(x),
                         kernel_size=dim),
            (bs, 1, -1)).softmax(dim=-1)
        xv = torch.reshape(self.wv(x), (bs, self._ch, -1))
        x_ = torch.matmul(xq, xv)
        return x_.reshape((bs, 1, *dim)).sigmoid() * x


class PSA(nn.Module):
    """ Polarized Self-Attention module from https://arxiv.org/abs/2107.00782
    """
    def __init__(
            self,
            in_channels: int,
            channel_factor: float = .5,
            mode: str = 'parallel'
    ):
        super(PSA, self).__init__()
        if mode not in ['parallel', 'sequential']:
            raise ValueError('PSA only support either `parallel` or `sequential` configuration.')

        self.psa_ch = PSAChannel(in_channels=in_channels, channel_factor=channel_factor)
        self.psa_sp = PSASpatial(in_channels=in_channels, channel_factor=channel_factor)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ch = self.psa_ch(x)
        return self.psa_sp(x_ch) if self.mode == 'sequential' else x_ch + self.psa_sp(x)


class PANet(nn.Module):
    def __init__(
            self,
            channels: List[int],
            shortcut: bool = True,
            n_c3: int = 3,
            expansion: float = .5
    ):
        super(PANet, self).__init__()
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        self.num_inputs = len(channels) // 2 + 1

        self.up_convs = nn.ModuleList()
        self.up_c3s = nn.ModuleList()

        for c_in, c_out in zip(channels[: len(channels) // 2],
                               channels[1: len(channels) // 2 + 1]):
            conv_bn = ConvBnAct(
                c_in=c_in,
                c_out=c_out,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            c3 = BottleneckC3SP(
                c_in=c_out * 2,
                c_out=c_out,
                n=n_c3,
                shortcut=shortcut,
                expansion=expansion
            )
            self.up_convs.append(conv_bn)
            self.up_c3s.append(c3)

        self.down_convs = nn.ModuleList()
        self.down_c3s = nn.ModuleList()
        for c_in, c_out in zip(channels[-(len(channels) // 2 + 1): -1],
                               channels[-len(channels) // 2 + 1:]):
            conv_bn = ConvBnAct(
                c_in=c_in,
                c_out=c_in,
                kernel_size=3,
                stride=2,
                padding=1
            )
            c3 = BottleneckC3SP(
                c_in=c_in * 2,
                c_out=c_out,
                n=n_c3,
                shortcut=shortcut,
                expansion=expansion
            )
            self.down_convs.append(conv_bn)
            self.down_c3s.append(c3)

    def forward(self, x) -> List[torch.Tensor]:
        assert len(x) == self.num_inputs
        init_x = x[0]
        output = []
        skip_connection = []

        for _x, conv, c3 in zip(x[1:], self.up_convs, self.up_c3s):
            init_x = conv(init_x)
            skip_connection.append(init_x)
            init_x = self.up_sample(init_x)
            init_x = torch.concat([init_x, _x], dim=1)
            init_x = c3(init_x)

        output.append(init_x)
        for _x, conv, c3 in zip(skip_connection[::-1], self.down_convs, self.down_c3s):
            init_x = conv(init_x)
            init_x = torch.concat([init_x, _x], dim=1)
            init_x = c3(init_x)
            output.append(init_x)

        return output


class KAPAOHead(nn.Module):
    def __init__(
            self,
            anchor: tuple,
            stride: float,
            ch_input: int,
            num_class: int = 18,
            num_coord: int = 34,
    ):
        super(KAPAOHead, self).__init__()
        self.num_class = num_class
        self.num_coord = num_coord
        self.num_output = self.num_class + 5 + self.num_coord
        self.num_anchor = len(anchor) // 2

        self.ch_input = ch_input

        self.register_buffer('stride', torch.tensor(stride, dtype=torch.float))

        anchor = torch.tensor(anchor, dtype=torch.float).view(-1, 2)
        self.register_buffer('anchor', anchor)
        self.register_buffer('anchor_grid', anchor.clone().view(1, -1, 1, 1, 2))

        self.grid = torch.zeros(1)

        self.conv_head = nn.Conv2d(ch_input, self.num_output * self.num_anchor, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.conv_head(x)
        bs, ch, h, w = x.shape
        x = x.view(bs, self.num_anchor, self.num_output, h, w).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return x

        if self.grid.shape[2:4] != x.shape[2:4]:
            self.grid = self._make_grid(w, h)

        y = x.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - .5 + self.grid) * self.stride
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid

        if self.num_coord:
            y[..., -self.num_coord:] = y[..., -self.num_coord:] * 4. - 2.
            y[..., -self.num_coord:] *= self.anchor_grid.repeat((1, 1, 1, 1, self.num_coord // 2))
            y[..., -self.num_coord:] += (self.grid * self.stride).repeat((1, 1, 1, 1, self.num_coord // 2))

        return x, y.view(bs, -1, self.num_output)

    def _make_grid(
            self,
            nx: int,
            ny: int,
    ):
        shape = 1, self.num_anchor, nx, ny, 2
        y = torch.arange(ny, device=self.anchor.device, dtype=self.anchor.dtype)
        x = torch.arange(nx, device=self.anchor.device, dtype=self.anchor.dtype)

        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([xv, yv], dim=2).expand(shape) - .5
        return grid


__all__ = [
    'ConvBnAct',
    'DepthWiseConvBnAct',
    'PointWiseConvBnAct',
    'Bottleneck',
    'BottleneckC3SP',
    'BottleneckCSP',
    'SpatialPyramidPooling',
    'SPP',
    'SpatialPyramidPoolingFast',
    'SPPF',
    'PSAChannel',
    'PSASpatial',
    'PSA',
    'PANet',
    'KAPAOHead',
    'Focus',
]


