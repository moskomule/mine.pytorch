import torch
from torch import nn, optim


class MINE(object):
    def __init__(self, static_net: nn.Module, lr: float, momentum: float):
        """
        Mutual Information Neural Estimation
        """
        self.static_net = static_net
        self.optimizer = optim.SGD(static_net.parameters(),
                                   lr=lr, momentum=momentum)
        self._is_train = True

    def __call__(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        calculate MI and if training mode, update the parameters
        """
        _z = z[torch.randperm(z.size(0))]
        mi = self.static_net(x, z).mean() - \
            self.static_net(x, _z).exp().mean().log()
#        print(
#            f"self.static_net(x, z).mean() {self.static_net(x, z).mean().item()}")
#        print(f"self.static_net(x, _z).exp().mean().log() {self.static_net(x, _z).exp().mean().log()}")
#        exit()

        if self._is_train:
            self.optimizer.zero_grad()
            (-mi).backward(retain_graph=True)
            self.optimizer.step()

        return mi

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False
