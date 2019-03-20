import math
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from gaussian import StaticNet


functions = {"x": lambda x: x,
             "x^3": lambda x: x**3,
             "sin(x)": lambda x: math.sin(x)}

def nonlin_data_loader(dim: int, num_samples: int, rho: float, batch_size: int, function) -> DataLoader:
    base = torch.empty(num_samples, dim).uniform_(-1, 1)
    y = function(base) + rho * base.clone().normal_()
    dataset = TensorDataset(base, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)


if __name__ == "__main__":
    from mine import MINE
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=128*20)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--function", default="x", choices=["x^3", "sin(x)", "x"])
    args = p.parse_args()

    loader = nonlin_data_loader(
        args.dim, args.num_samples, args.rho, args.batch_size, functions[args.function])
    static_net = StaticNet(args.dim, args.hidden)
    m = MINE(static_net, lr=args.lr, momentum=0.9)
    for ep in range(200):
        mi = 0
        counter = 0
        for x, z in loader:
            mi += m(x, z).item()
            counter += 1

        print(f"epoch {ep:>4} MI: {mi/counter:.3f}")
