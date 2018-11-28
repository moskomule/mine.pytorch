import numpy as np
from torch import nn, Tensor, cat
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F


class StaticNet(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_size)
        self.linear2 = nn.Linear(dim, hidden_size)
        self.linear3 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = F.leaky_relu(self.linear1(x))
        z = F.leaky_relu(self.linear2(z))
        x = F.leaky_relu(self.linear3(cat([x, z], dim=1)))
        return self.linear4(x)


def gaussian_data_loader(dim: int, num_samples: int, rho: float, batch_size: int) -> DataLoader:
    size = 2 * dim
    base = np.eye(size)
    base[range(size), [(i+dim) % size for i in range(size)]] = rho
    random = Tensor(np.random.multivariate_normal(
        np.zeros(size), base, num_samples))
    dataset = TensorDataset(random[:, :dim], random[:, dim:])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    from mine import MINE
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=128*20)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=16)
    args = p.parse_args()

    loader = gaussian_data_loader(
        args.dim, args.num_samples, args.rho, args.batch_size)
    static_net = StaticNet(args.dim, args.hidden)
    m = MINE(static_net, lr=args.lr, momentum=0.9)
    for ep in range(200):
        mi = 0
        counter = 0
        for x, z in loader:
            mi += m(x, z).item()
            counter += 1

        print(f"epoch {ep:>4} MI: {mi/counter:.3f}")
