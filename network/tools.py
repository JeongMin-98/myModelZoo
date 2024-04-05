""" Tools for implementing neural networks """
from torch import nn
from torch.nn import Conv2d
from torchvision.ops import MLP

from network.module import NiNBlock, InceptionSequential


# def _add_inception_block(previous_out_dim, block_info):
def _add_inception_block(block_info):
    params = {
        # "in_channel": previous_out_dim,
        "in_channel": int(block_info["in_channel"]),
        "conv1x1_out_channels": block_info["conv1x1_out_channels"],
        "conv3x3_reduce_out_channels": block_info["conv3x3_reduce_out_channels"],
        "conv3x3_out_channels": block_info["conv3x3_out_channels"],
        "conv5x5_reduce_out_channels": block_info["conv5x5_reduce_out_channels"],
        "conv5x5_out_channels": block_info["conv5x5_out_channels"],
        "proj_out_channels": block_info["proj_out_channels"],
        "depth": int(block_info["depth"])
    }
    return InceptionSequential(**params)


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
        "dropout": float(block_info["dropout"]) if float(block_info["dropout"]) > 0 else 0.0,
        "activation_layer": nn.ReLU if block_info["activation_layer"] == "ReLU" else nn.Tanh
    }
    block = MLP(**params)
    return block


def _add_linear(block_info):
    params = {
        'in_features': int(block_info["in_channels"]),
        'out_features': int(block_info['out_channels'])
    }
    return nn.Linear(**params)


def _add_conv_block(block_info):
    params = {
        "in_channels": int(block_info["in_channels"]),
        "out_channels": int(block_info["out_channels"]),
        "kernel_size": int(block_info["kernel_size"]),
        "stride": int(block_info["stride"]),
        "padding": int(block_info.get("padding", 0)),  # Default padding is 0 if not specified
        "padding_mode": "replicate"
    }

    return Conv2d(**params)


def _add_pooling_layer(block_info):
    if block_info["method"] == "AdaptAvg":
        return nn.AdaptiveAvgPool2d((1, 1))
    pooling_layer = nn.AvgPool2d if block_info["method"] == "average" else nn.MaxPool2d
    params = {
        "kernel_size": int(block_info["kernel_size"]),
        "stride": int(block_info["stride"]),
        "padding": int(block_info["padding"])
    }

    return pooling_layer(**params)


def _add_dropout(block_info):
    params = {
        "p": float(block_info["dropout_ratio"])
    }
    return nn.Dropout(**params)


def set_layer(config):
    """ set layer from config file """
    module_list = nn.ModuleList()
    # top_dim = None
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
        if info['type'] == 'Linear':
            module_list.append(_add_linear(info))
        if info['type'] == 'Inception':
            module_list.append(_add_inception_block(info))
        if info['type'] == 'dropout':
            module_list.append(_add_dropout(info))

        if "activation_layer" in info.keys():
            if info['activation_layer'] == "relu":
                module_list.append(nn.ReLU(inplace=True))
            if info['activation_layer'] == "tanh":
                module_list.append(nn.Tanh())
    return module_list
