"""Build the figure the README shows: one panel per capability.

Run it as ``python examples/figures/make_showcase.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchsolve import Regularizer, conjugate_gradient


def badly_scaled_system(size=512):
    """An operator whose scale spans three decades, as sampling density does."""
    generator = torch.Generator().manual_seed(0)
    density = torch.logspace(0, 3, size, dtype=torch.float64)
    coupling = torch.randn(size, size, generator=generator, dtype=torch.float64)
    coupling = 0.35 * (coupling + coupling.T) / size**0.5
    matrix = torch.diag(density) + coupling * density.sqrt().outer(density.sqrt())
    truth = torch.randn(size, generator=generator, dtype=torch.float64)
    return (lambda vector: matrix @ vector), density, truth, matrix @ truth


def residual_history(normal, target, truth, preconditioner, iterations):
    """The error after each iteration count, solved afresh each time."""
    return [
        (
            conjugate_gradient(
                normal, target, max_iter=count, preconditioner=preconditioner
            ).solution
            - truth
        )
        .norm()
        .item()
        / truth.norm().item()
        for count in range(1, iterations + 1)
    ]


STAIRCASE = 200


def noisy_staircase(size=STAIRCASE):
    """A piecewise-constant signal, measured with noise."""
    generator = torch.Generator().manual_seed(0)
    truth = torch.zeros(size, dtype=torch.float64)
    for start, stop, level in ((20, 70, 1.0), (70, 120, -0.5), (140, 180, 0.7)):
        truth[start:stop] = level
    measured = truth + 0.35 * torch.randn(
        size, generator=generator, dtype=torch.float64
    )
    return truth, measured


LEARNED = 120


def problem():
    """A rough truth, a noisy measurement of it, and the penalty operator."""
    generator = torch.Generator().manual_seed(0)
    index = torch.arange(LEARNED, dtype=torch.float64)
    truth = torch.sin(index / 9.0) + 0.4 * torch.sin(index / 2.5)
    measured = truth + 0.4 * torch.randn(
        LEARNED, generator=generator, dtype=torch.float64
    )
    difference = torch.diff(torch.eye(LEARNED, dtype=torch.float64), dim=0)
    return truth, measured, difference


plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})


def main() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    normal, density, truth, target = badly_scaled_system()
    iterations = 40
    plain = residual_history(normal, target, truth, None, iterations)
    fixed = residual_history(normal, target, truth, lambda v: v / density, iterations)
    steps = range(1, iterations + 1)
    axes[0].semilogy(steps, plain, label="plain")
    axes[0].semilogy(steps, fixed, label="preconditioned")
    axes[0].set_title("preconditioning, not density weighting")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("relative error")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    reference, measured = noisy_staircase()
    difference = torch.diff(torch.eye(STAIRCASE, dtype=torch.float64), dim=0)
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

    true_signal, noisy, penalty = problem()
    grid = torch.logspace(-2, 2, 30, dtype=torch.float64)
    swept = []
    for candidate in grid:
        estimate = conjugate_gradient(
            lambda v, c=candidate: v + c * (penalty.T @ (penalty @ v)),
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
            lambda v, w=weight: v + w * (penalty.T @ (penalty @ v)),
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
    axes[2].semilogx(
        weights, losses, "o-", markersize=2.5, linewidth=0.7, label="learned"
    )
    axes[2].set_title("differentiable, without unrolling")
    axes[2].set_xlabel("regularisation weight")
    axes[2].set_ylabel("mean squared error")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    destination = Path(__file__).parent / "showcase.png"
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    print("wrote", destination)


if __name__ == "__main__":
    main()
