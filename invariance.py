import torch
from homura.optim import Adam
from torch.distributions import MultivariateNormal

from modules import KLMINE, FCStaticNet


def normalize(t: torch.Tensor):
    return t / t.norm(dim=-1, keepdim=True)


def main():
    final_results = {}
    function = {"x": lambda x: x,
                "x^3": lambda x: x ** 3,
                "sin(x)": lambda x: x.sin()}[args.function]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for rho in args.rho:
        rho = rho / 10
        final_results[rho] = []
        mine = KLMINE(FCStaticNet(2, 2), Adam())
        mine.to(device)
        gaussian = MultivariateNormal(torch.zeros(2, device=device), torch.eye(2, device=device))
        for ep in range(args.epochs):
            input = (torch.rand(args.batch_size, 2) * 2 - 1).to(device)
            target = function(input) + rho * gaussian.sample(torch.Size([512]))
            mi = mine(normalize(input), normalize(target))
            if torch.isnan(mi):
                print(f">>> {rho:2>}/{ep}, ")
            final_results[rho].append(mi.item())

        print(f"{rho:>5}={mi.item():.4f}")


if __name__ == '__main__':
    import miniargs

    p = miniargs.ArgumentParser()
    p.add_int("--epochs", default=10_000)
    p.add_int("--batch_size", default=512)
    p.add_multi_float("--rho", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.9])
    p.add_str("--function", choices=["x", "x^3", "sin(x)"])

    args = p.parse()
    main()
