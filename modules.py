import math
import torch
from torch import nn


def pair(x):
    if len(tuple(x)) == 1:
        return (x, x)
    else:
        return x


class BiasConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        """
        >>> conv = BiasConv2d(32, 32, 3)
        >>> conv(torch.randn(32, 32, 28, 28), bias)
        """

        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels // groups, *self.kernel_size))

    def forward(self, input, bias):
        return F.conv2d(input, self.weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
