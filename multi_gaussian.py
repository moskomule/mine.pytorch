import torch
from homura.optim import Adam
from torch.distributions import MultivariateNormal

from modules import MINE, FCStaticNet


def main():
    final_results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for rho in args.rho:
        rho = rho / 10
        final_results[rho] = []
        mine = MINE(FCStaticNet(1, 1), Adam())
        mine.to(device)
        gaussian = MultivariateNormal(torch.zeros(2),
                                      covariance_matrix=torch.tensor([[1, rho], [rho, 1]]))
        data = gaussian.sample(torch.Size([512])).t().to(device)
        for ep in range(args.epochs):
            mi = mine(data[0].view(-1, 1), data[1].view(-1, 1))
            final_results[rho].append(mi.item())

        print(f"{rho:>5}={mi.item():.4f}")


if __name__ == '__main__':
    import miniargs

    p = miniargs.ArgumentParser()
    p.add_int("--epochs", default=5000)
    p.add_multi_float("--rho", default=[-9.9, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.9])

    args = p.parse()
    main()
