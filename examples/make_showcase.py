"""Build the figure the README shows: one panel per capability."""

import matplotlib.pyplot as plt
import torch

import differentiable_weight as learned
import preconditioning as conditioning
import regularised_solve as shaped
from torchsolve import Regularizer, conjugate_gradient

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})


def main() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    normal, density, truth, target = conditioning.badly_scaled_system()
    iterations = 40
    plain = conditioning.residual_history(normal, target, truth, None, iterations)
    fixed = conditioning.residual_history(
        normal, target, truth, lambda v: v / density, iterations
    )
    steps = range(1, iterations + 1)
    axes[0].semilogy(steps, plain, label="plain")
    axes[0].semilogy(steps, fixed, label="preconditioned")
    axes[0].set_title("preconditioning, not density weighting")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("relative error")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    reference, measured = shaped.noisy_staircase()
    difference = torch.diff(torch.eye(shaped.SIZE, dtype=torch.float64), dim=0)
    smoothed = conjugate_gradient(
        lambda v: v,
        measured,
        regularizers=[
            Regularizer(
                8.0,
                operator=lambda v: difference @ v,
                adjoint=lambda v: difference.T @ v,
            )
        ],
        max_iter=400,
        rtol=1e-12,
    ).solution
    axes[1].plot(measured, linewidth=0.7, alpha=0.45, label="measured")
    axes[1].plot(reference, linewidth=1.3, label="truth")
    axes[1].plot(smoothed, linewidth=1.8, label="with a shaped penalty")
    axes[1].set_title("arbitrary regularisation operators")
    axes[1].set_xlabel("sample")
    axes[1].legend()

    true_signal, noisy, penalty = learned.problem()
    grid = torch.logspace(-2, 2, 30, dtype=torch.float64)
    swept = []
    for candidate in grid:
        estimate = conjugate_gradient(
            lambda v: v + candidate * (penalty.T @ (penalty @ v)),
            noisy,
            max_iter=200,
            rtol=1e-12,
        ).solution
        swept.append((estimate - true_signal).pow(2).mean().item())
    log_weight = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([log_weight], lr=0.25)
    path = []
    for _ in range(60):
        optimiser.zero_grad()
        weight = log_weight.exp()
        estimate = conjugate_gradient(
            lambda v: v + weight * (penalty.T @ (penalty @ v)),
            noisy,
            max_iter=200,
            rtol=1e-12,
            parameters=[log_weight],
        ).solution
        loss = (estimate - true_signal).pow(2).mean()
        loss.backward()
        optimiser.step()
        path.append((weight.item(), loss.item()))
    weights, losses = zip(*path, strict=True)
    axes[2].semilogx(grid, swept, linewidth=1.2, label="swept")
    axes[2].semilogx(weights, losses, "o-", markersize=2.5, linewidth=0.7,
                     label="learned")
    axes[2].set_title("differentiable, without unrolling")
    axes[2].set_xlabel("regularisation weight")
    axes[2].set_ylabel("mean squared error")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig("figures/showcase.png", dpi=140, bbox_inches="tight")
    print("wrote figures/showcase.png")


if __name__ == "__main__":
    main()
