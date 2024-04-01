import torch
import torch.nn as nn
from typing import List, Optional, Callable


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
