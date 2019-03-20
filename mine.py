import torch
from torch import nn, optim


class MINE(object):
    def __init__(self, static_net: nn.Module, lr: float, momentum: float, eps=1e-4):
        """
        Mutual Information Neural Estimation
        """
        self.static_net = static_net
        self.optimizer = optim.SGD(static_net.parameters(),
                                   lr=lr, momentum=momentum, weight_decay=1e-4)
        self._is_train = True
        self._eps = eps

    def __call__(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        calculate MI and if training mode, update the parameters
        """
        _z = z[torch.randperm(z.size(0))]
        mi = self.static_net(x, z).mean() - \
            (self.static_net(x, _z).exp().mean()+self._eps).log()
        if torch.isnan(mi).sum().item() > 0:
            print(self.static_net(x, z))
            print((self.static_net(x, _z).exp().mean()+self._eps).log())
            exit(1)

        if self._is_train:
            self.optimizer.zero_grad()
            (-mi).backward(retain_graph=True)
            self.optimizer.step()

        return mi

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False
