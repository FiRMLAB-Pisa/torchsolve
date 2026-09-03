"""What one linearised step hands to whichever solver is doing the work."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from ._cg import Regularizer

__all__ = ["InnerSolver", "LinearProblem"]

Operator = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class LinearProblem:
    r"""The regularised least-squares problem of one Gauss-Newton step.

    .. math::

        \min_z \; \|J z - r\|^2
            + \alpha \sum_k w_k \|R_k z - c_k\|^2

    Carries the problem in both of the forms a solver might want it: as the
    normal operator and its right-hand side, which is all an iterative solver
    needs and all a matrix-free one can have, and as the Jacobian and the
    residual themselves, which a direct solver wants because stacking is
    better conditioned than squaring.

    Parameters
    ----------
    normal
        :math:`J^H J`.
    rhs
        :math:`J^H r`.
    alpha
        The step's regularisation strength. Every regularizer is scaled by it,
        so a single scalar governs them all -- which is what makes the
        Gauss-Newton schedule a schedule.
    regularizers
        Terms, whose own weights are *relative* to ``alpha``. The default,
        an empty tuple, means one identity term at weight one: plain Tikhonov
        towards the reference, which is what IRGNM asks for.
    matrix
        :math:`J`, when it is a dense matrix rather than an operator.
    target
        :math:`r`, in data space. Present whenever ``matrix`` is.
    x0
        Starting iterate, for a solver that can use one.
    """

    normal: Operator
    rhs: torch.Tensor
    alpha: float
    regularizers: Sequence[Regularizer] = ()
    matrix: torch.Tensor | None = field(default=None, repr=False)
    target: torch.Tensor | None = field(default=None, repr=False)
    x0: torch.Tensor | None = field(default=None, repr=False)

    def scaled(self) -> tuple[Regularizer, ...]:
        """Return the regularizers with ``alpha`` folded into their weights."""
        if not self.regularizers:
            return (Regularizer(self.alpha),)
        return tuple(
            Regularizer(
                weight=self.alpha * term.weight,
                operator=term.operator,
                adjoint=term.adjoint,
                bias=term.bias,
            )
            for term in self.regularizers
        )


@runtime_checkable
class InnerSolver(Protocol):
    """Anything that can solve a :class:`LinearProblem`.

    The interface is one call, so a solver this package does not provide --
    ADMM, a proximal method, something learned -- can be handed to
    :func:`torchsolve.gauss_newton` without this package knowing about it.
    """

    def __call__(self, problem: LinearProblem) -> torch.Tensor:
        """Return the step that solves ``problem``."""
        ...
