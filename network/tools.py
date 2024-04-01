""" Tools for implementing neural networks """
from torch import nn
from torch.nn import Conv2d
from torchvision.ops import MLP

from network.module import NiNBlock


def _add_nin_block(block_info):
    params = {
        "in_channels": int(block_info["in_channels"]),
        "out_channels": block_info["out_channels"],
        "kernel_size": int(block_info["kernel_size"]),
        "depth": int(block_info["depth"]),
        "stride": int(block_info["stride"]),
        "padding": int(block_info["padding"]),
        "activation_layer": nn.ReLU if block_info["activation_layer"] == "ReLU" else nn.Tanh,
    }

    return NiNBlock(**params)


def _add_mlp_block(block_info):
    params = {
        "in_channels": int(block_info["in_channels"]),
        "hidden_channels": block_info["hidden_channels"],
        "dropout": float(block_info["dropout"]),
        "activation_layer": nn.ReLU if block_info["activation_layer"] == "ReLU" else nn.Tanh
    }
    block = MLP(**params)
    return block


def _add_conv_block(block_info):
    params = {
        "in_channels": int(block_info["in_channels"]),
        "out_channels": int(block_info["out_channels"]),
        "kernel_size": int(block_info["kernel_size"]),
        "stride": int(block_info["stride"]),
        "padding": int(block_info.get("padding", 0))  # Default padding is 0 if not specified
    }
    return Conv2d(**params)


def _add_pooling_layer(block_info):
    pooling_layer = nn.AvgPool2d if block_info["method"] == "average" else nn.MaxPool2d

    params = {
        "kernel_size": int(block_info["kernel_size"]),
        "stride": int(block_info["stride"])
    }

    return pooling_layer(**params)


def set_layer(config):
    """ set layer from config file """
    module_list = nn.ModuleList()
    # sequential_layers = []
    # input shape

    # iter config files
    for idx, info in enumerate(config):
        if info['type'] == 'MLP':
            module_list.append(_add_mlp_block(info))
            continue
        if info['type'] == 'NiN':
            module_list.append(_add_nin_block(info))
            continue
        if info['type'] == 'Output':
            module_list.append(nn.LogSoftmax(dim=1))
        if info['type'] == 'Conv':
            module_list.append(_add_conv_block(info))
        if info['type'] == 'Pooling':
            module_list.append(_add_pooling_layer(info))
        if info['type'] == 'Flatten':
            module_list.append(nn.Flatten())

        if "activation_layer" in info.keys():
            if info['activation_layer'] == "relu":
                module_list.append(nn.ReLU(inplace=True))
            if info['activation_layer'] == "tanh":
                module_list.append(nn.Tanh())
    return module_list
