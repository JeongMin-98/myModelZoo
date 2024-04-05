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
    def __init__(self):
        super(Inception, self).__init__()
        prev_channels = 192
        self.branch1 = nn.Sequential(
            nn.Conv2d(prev_channels, 64, kernel_size=(1, 1))
        )  # 1*1 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(prev_channels, 96, kernel_size=(1, 1)),
            nn.Conv2d(96, 128, kernel_size=(3, 3))
        )  # 1*1 conv(reduce) + 3*3 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(prev_channels, 16, kernel_size=(1, 1)),
            nn.Conv2d(16, 32, kernel_size=(5, 5))
        )  # 1*1 conv(reduce) + 5*5 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d((3, 3), stride=1),
            nn.Conv2d(prev_channels, 32, kernel_size=(1, 1))
        )  # maxpool(3*3) + 1*1 conv

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4])
        return x


if __name__ == '__main__':
    # write test code
    pass
