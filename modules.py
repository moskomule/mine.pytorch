import math

import torch
from homura.optim import SGD, Optimizer
from torch import nn
from torch.nn import functional as F


def pair(i):
    if len(tuple(i)) == 1:
        return (i, i)
    return i


class BiasConv2d(nn.Module):
    # conv net which takes input and class conditional bias
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 num_classes: int,
                 stride=1, padding=0, dilation=1, groups=1):
        super(BiasConv2d, self).__init__()
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.weight = nn.Parameter(torch.empty(in_channels,
                                               out_channels // groups,
                                               *self.kernel_size))
        self.bias = nn.Linear(num_classes, out_channels)

    def forward(self,
                input: torch.Tensor,
                bias: torch.Tensor):
        return F.conv2d(input, self.weight, self.bias(bias), self.stride, self.padding,
                        self.dilation, self.groups)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, std: float = 1):
        super(GaussianNoiseLayer, self).__init__()
        self.std = std

    def forward(self, input: torch.Tensor):
        return input + torch.empty_like(input).normal_(0, self.std)


class ConcatLayer(nn.Module):
    def __init__(self,
                 cat_dim=-1):
        super(ConcatLayer, self).__init__()
        self.cat_dim = cat_dim

    def forward(self, *input):
        return torch.cat(input, dim=self.cat_dim)


def fc_static_net(x_dim: int,
                  z_dim: int,
                  hidden_dim: int = 512):
    return nn.Sequential(ConcatLayer(),
                         GaussianNoiseLayer(0.3),
                         nn.Linear(x_dim + z_dim, hidden_dim),
                         nn.ELU(),
                         nn.Linear(hidden_dim, hidden_dim),
                         nn.ELU(),
                         nn.Linear(hidden_dim, 1))


class ConvStaticNet(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super(ConvStaticNet, self).__init__()
        self.conv1 = BiasConv2d(3, 16, 5, num_classes, stride=2, padding=2)
        self.conv2 = BiasConv2d(16, 32, 5, num_classes, stride=2, padding=2)
        self.conv3 = BiasConv2d(32, 64, 5, num_classes, stride=2, padding=2)
        self.conv4 = BiasConv2d(64, 128, 5, num_classes, stride=2, padding=2)
        self.linear = nn.Linear(input_size // 16, 1)

    def forward(self,
                input: torch.Tensor,
                bias: torch.Tensor):
        x = F.elu(self.conv1(input, bias))
        x = F.elu(self.conv2(x, bias))
        x = F.elu(self.conv3(x, bias))
        x = F.elu(self.conv4(x, bias))
        return self.linear(F.adaptive_avg_pool2d(x, 1))


class MINE(object):
    """
    >>> mine = MINE(fc_static_net(10, 10,))
    >>> mine.to("cuda")
    >>> mine(torch.randn(4, 10), torch.randn(4, 10))
    >>> mine.eval()
    """

    def __init__(self,
                 static_net: nn.Module,
                 optimizer: Optimizer = None):
        self.static_net = static_net
        _optimizer = optimizer
        if optimizer is None:
            _optimizer = SGD(lr=0.01, momentum=0.9)
        self._optimizer = _optimizer
        self.optimizer = optimizer.set_model(self.static_net)
        self.training = True

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # returns lower bound of MI
        # when update, update the negative of the returned value
        joint = self.static_net(input, target).mean(dim=-1)
        # -log(size) for averaging
        margin = self.static_net(input, target[torch.randperm(target.size(0))]).logsumexp(dim=-1) \
                 - math.log(target.size(0))
        return joint + margin

    def __call__(self,
                 input: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        mi = self.forward(input, target)
        if not self.training:
            return mi
        self.optimizer.zero_grad()
        (-mi).backward()
        self.optimizer.step()
        return mi.detach()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def to(self,
           device: torch.device):
        self.static_net.to(device)
        self.optimizer = self._optimizer.set_model(self.static_net)
