"""The general form: a data term, a shaped penalty, and a bias.

The solver takes any number of terms of the form ``weight * ||R x - c||^2``. R
defaults to the identity, which pulls towards c, and c defaults to zero, which
pulls towards the origin. A first-difference R penalises roughness instead.
"""

import matplotlib.pyplot as plt
import torch

from torchsolve import Regularizer, conjugate_gradient

SIZE = 200


def noisy_staircase(size: int = SIZE):
    """A piecewise-constant signal, measured with noise."""
    generator = torch.Generator().manual_seed(0)
    truth = torch.zeros(size, dtype=torch.float64)
    for start, stop, level in ((20, 70, 1.0), (70, 120, -0.5), (140, 180, 0.7)):
        truth[start:stop] = level
    measured = truth + 0.35 * torch.randn(size, generator=generator, dtype=torch.float64)
    return truth, measured


def main() -> None:
    truth, measured = noisy_staircase()
    identity = lambda vector: vector
    difference = torch.diff(torch.eye(SIZE, dtype=torch.float64), dim=0)

    plain = conjugate_gradient(identity, measured, max_iter=200, rtol=1e-12).solution
    smoothed = conjugate_gradient(
        identity,
        measured,
        regularizers=[
            Regularizer(
                8.0,
                operator=lambda vector: difference @ vector,
                adjoint=lambda vector: difference.T @ vector,
            )
        ],
        max_iter=400,
        rtol=1e-12,
    ).solution
    # A bias relocates what the term pulls towards. With none it pulls to the
    # origin, which shrinks the answer; with one it pulls to that instead.
    prior = torch.full((SIZE,), 2.0, dtype=torch.float64)
    shrunk = conjugate_gradient(
        identity, measured, regularizers=[Regularizer(4.0)], max_iter=200, rtol=1e-12
    ).solution
    anchored = conjugate_gradient(
        identity,
        measured,
        regularizers=[Regularizer(4.0, bias=prior)],
        max_iter=200,
        rtol=1e-12,
    ).solution
    print(f"the same weight pulls to zero (mean {shrunk.mean():+.3f}) or to a "
          f"prior of 2.0 (mean {anchored.mean():+.3f})")

    error = lambda estimate: (estimate - truth).norm().item() / truth.norm().item()
    print(f"relative error: measured {error(measured):.3f}, "
          f"unregularised {error(plain):.3f}, smoothed {error(smoothed):.3f}")

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(measured, linewidth=0.8, alpha=0.5, label="measured")
    axis.plot(truth, linewidth=1.5, label="truth")
    axis.plot(plain, linewidth=0.8, alpha=0.7, label="no penalty")
    axis.plot(smoothed, linewidth=2.0, label="first-difference penalty")
    axis.set_xlabel("sample")
    axis.legend(fontsize=8)
    axis.set_title("one data term, one shaped penalty")
    figure.tight_layout()
    figure.savefig("figures/regularised_solve.png", dpi=120)
    print("wrote figures/regularised_solve.png")


if __name__ == "__main__":
    main()
