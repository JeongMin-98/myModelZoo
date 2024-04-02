from torch import nn
from network.tools import set_layer

""" Neural Network template (Using config file) """


class Net(nn.Module):
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.layers = set_layer(self.config)
        self.Linear = nn.Linear(256, 10)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear(x)
        return x
