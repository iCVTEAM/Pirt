
import torch
import torch.nn as nn
from copy import deepcopy

class MLP(nn.Module):
    def __init__(self, channels: list, do_bn=True, xd='1'):
        super(MLP, self).__init__()
        n = len(channels)
        layers = []
        conv = nn.Conv1d if xd == '1' else nn.Conv2d
        bn = nn.InstanceNorm1d if xd == '1' else nn.InstanceNorm2d
        for i in range(1, n):
            layers.append(
                conv(channels[i - 1], channels[i], kernel_size=1, bias=True)
            )
            if i < n - 1:
                if do_bn:
                    layers.append(bn(channels[i]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def MLP(channels: list, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        )
        if i < n-1:
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
