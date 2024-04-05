import torch
import torch.nn as nn
from typing import List, Optional, Callable
from collections import OrderedDict


class NiNBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 kernel_size: int,
                 stride: int,
                 activation_layer: Optional[Callable[..., nn.Module]] = torch.nn.ReLU,
                 depth: int = 2,
                 padding=0,
                 inplace=True
                 ):

        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel_size of first conv layer
        :param stride: stride of first conv layer
        :param padding: padding of first conv layer
        :param depth: depth of MLPconv (number of 1*1 conv layer)
        """

        params = {} if inplace is None else {"inplace": inplace}
        layers = []

        if len(out_channels) < 1:
            raise ValueError("out_channels must not be greater than 1")

        if len(out_channels) > depth:
            raise ValueError("out_channels must be less than depth")

        in_dim = in_channels

        layers.append(nn.Conv2d(in_dim, out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(activation_layer(**params))

        in_dim = out_channels[0]

        if len(out_channels) == 1:
            for d in range(depth - 1):
                layers.append(nn.Conv2d(in_dim, out_channels[0], kernel_size=1))
                layers.append(activation_layer(**params))
        elif len(out_channels) >= 2:

            for out_channel in out_channels[:-1]:
                layers.append(nn.Conv2d(in_dim, out_channel, kernel_size=1))
                layers.append(activation_layer(**params))
                in_dim = out_channel

        layers.append(nn.Conv2d(in_dim, out_channels[-1], kernel_size=1))
        layers.append(activation_layer(**params))

        super().__init__(*layers)


class Inception(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv1x1_out_channels: int,
                 conv3x3_reduce_out_channels: int,
                 conv3x3_out_channels: int,
                 conv5x5_reduce_out_channels: int,
                 conv5x5_out_channels: int,
                 proj_out_channels: int,
                 ):
        """
        :param in_channels: number of previous channels (input channels)
        """
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(OrderedDict([
            ('1x1 conv', nn.Conv2d(in_channels, conv1x1_out_channels, kernel_size=(1, 1))),
            ('ReLU', nn.ReLU(inplace=True))
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('3x3 Reduce', nn.Conv2d(in_channels, conv3x3_reduce_out_channels, kernel_size=(1, 1))),
            ('ReLU', nn.ReLU(inplace=True)),
            ('3x3 conv', nn.Conv2d(conv3x3_reduce_out_channels, conv3x3_out_channels, kernel_size=(3, 3), padding=1)),
            ('ReLU', nn.ReLU(inplace=True))
        ]))  # 1*1 conv(reduce) + 3*3 conv
        self.branch3 = nn.Sequential(OrderedDict([
            ('5x5 Reduce', nn.Conv2d(in_channels, conv5x5_reduce_out_channels, kernel_size=(1, 1))),
            ('ReLU', nn.ReLU(inplace=True)),
            ('5x5 conv', nn.Conv2d(conv5x5_reduce_out_channels, conv5x5_out_channels, kernel_size=(5, 5), padding=2)),
            ('ReLU', nn.ReLU(inplace=True)),
        ]))  # 1*1 conv(reduce) + 5*5 conv
        self.branch4 = nn.Sequential(OrderedDict([
            ('maxpool', nn.MaxPool2d((3, 3), stride=1, padding=1)),
            ('proj conv', nn.Conv2d(in_channels, proj_out_channels, kernel_size=(1, 1))),
            ('ReLU', nn.ReLU(inplace=True)),
        ]))  # maxpool(3*3) + 1*1 conv

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        concat_x = torch.cat([x1, x2, x3, x4], dim=1)
        return concat_x


class InceptionSequential(nn.Sequential):
    def __init__(self,
                 in_channel: int,
                 conv1x1_out_channels: List[int],
                 conv3x3_reduce_out_channels: List[int],
                 conv3x3_out_channels: List[int],
                 conv5x5_reduce_out_channels: List[int],
                 conv5x5_out_channels: List[int],
                 proj_out_channels: List[int],
                 depth=2
                 ):
        in_dim = in_channel

        layers = []
        params = []
        for i in range(depth):
            params = [
                in_dim,
                conv1x1_out_channels[i],
                conv3x3_reduce_out_channels[i],
                conv3x3_out_channels[i],
                conv5x5_reduce_out_channels[i],
                conv5x5_out_channels[i],
                proj_out_channels[i]
            ]
            next_dim = params[1] + params[3] + params[5] + params[6]
            layers.append(Inception(*params))
            in_dim = next_dim

        super().__init__(*layers)


if __name__ == '__main__':
    model = InceptionSequential(192, [64, 128], [96, 128], [128, 192], [16, 32], [32, 96], [32, 64])

    ipt = torch.randn(192, 28, 28)

    opt = model.forward(ipt)

    print(opt.shape)
