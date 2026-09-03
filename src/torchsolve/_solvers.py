"""Inner solvers: one iterative, one direct.

Which to use follows from the size of the step, not from taste. An iterative
solver needs only to apply the normal operator, so it is what a matrix-free
problem has; a direct one factorises the Jacobian, which is faster and exact
when the Jacobian is small enough to write down -- a per-voxel model fit, where
there are millions of tiny independent problems rather than one large one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._cg import Regularizer, conjugate_gradient
from ._problem import LinearProblem

__all__ = ["CGSolver", "LstsqSolver"]


@dataclass
class CGSolver:
    """Solve each step by conjugate gradient.

    Parameters
    ----------
    max_iter
        Iterations per step.
    rtol, atol
        Stopping tolerance, as in :func:`torchsolve.conjugate_gradient`.
    preconditioner
        Applied to the residual each iteration.
    batch_dim
        Axis along which the step is a batch of independent systems.
    warm_start
        Whether to begin each step from the previous one's answer. Off by
        default, because the regularisation changes between steps and the
        previous answer solved a different problem.

    Examples
    --------
    >>> from torchsolve import CGSolver
    >>> CGSolver(max_iter=30).max_iter
    30
    """

    max_iter: int = 100
    rtol: float = 0.0
    atol: float = 0.0
    preconditioner: object | None = None
    batch_dim: int | None = None
    warm_start: bool = False
    needs_matrix: bool = False

    def __call__(self, problem: LinearProblem) -> torch.Tensor:
        """Solve one step."""
        return conjugate_gradient(
            problem.normal,
            problem.rhs,
            x0=problem.x0 if self.warm_start else None,
            regularizers=problem.scaled(),
            preconditioner=self.preconditioner,  # type: ignore[arg-type]
            max_iter=self.max_iter,
            rtol=self.rtol,
            atol=self.atol,
            batch_dim=self.batch_dim,
            warn_indefinite=False,
        ).solution


@dataclass
class LstsqSolver:
    r"""Solve each step directly, by stacking the regularisers onto the Jacobian.

    .. math::

        \\min_z \\Big\\| \\begin{bmatrix} J \\\\ \\sqrt{\\alpha w_1} R_1 \\\\
        \\vdots \\end{bmatrix} z
        - \\begin{bmatrix} r \\\\ \\sqrt{\\alpha w_1} c_1 \\\\ \\vdots
        \\end{bmatrix} \\Big\\|^2

    Stacking rather than forming :math:`J^H J + \\alpha R^H R` squares nothing,
    so the conditioning the factorisation sees is that of the problem rather
    than its square. This is what generalises :func:`torch.linalg.lstsq`: the
    same solve, with any number of regularisation operators, their own relative
    weights and their own biases.

    Leading axes of the Jacobian are a batch, so a per-voxel fit solves every
    voxel in one call.

    Parameters
    ----------
    driver
        Passed to :func:`torch.linalg.lstsq`. ``None`` lets torch choose:
        ``gelsy`` on the host, ``gels`` on a device.

    Examples
    --------
    >>> from torchsolve import LstsqSolver
    >>> LstsqSolver().driver is None
    True
    """

    driver: str | None = None
    needs_matrix: bool = True

    def __call__(self, problem: LinearProblem) -> torch.Tensor:
        """Solve one step."""
        if problem.matrix is None or problem.target is None:
            raise ValueError(
                "a direct solve needs the Jacobian and the residual themselves, "
                "not only the normal operator; pass matrix= and target=, or use "
                "CGSolver"
            )
        jacobian, target = problem.matrix, problem.target
        rows, columns = jacobian.shape[-2], jacobian.shape[-1]
        blocks = [jacobian]
        pieces = [target]
        for term in problem.scaled():
            scale = term.weight**0.5
            if term.is_identity:
                eye = torch.eye(columns, dtype=jacobian.dtype, device=jacobian.device)
                block = eye.expand(*jacobian.shape[:-2], columns, columns) * scale
            else:
                block = _as_matrix(term, columns, jacobian) * scale
            blocks.append(block)
            bias = term.bias
            if bias is None:
                pieces.append(
                    torch.zeros(
                        *block.shape[:-1], dtype=target.dtype, device=target.device
                    )
                )
            else:
                pieces.append(bias * scale)
        stacked = torch.cat(blocks, dim=-2)
        answer = torch.cat(pieces, dim=-1)
        solution = torch.linalg.lstsq(
            stacked, answer.unsqueeze(-1), driver=self.driver
        ).solution
        del rows
        return solution.squeeze(-1)


def _as_matrix(term: Regularizer, columns: int, like: torch.Tensor) -> torch.Tensor:
    """Return the regularizer's operator as a matrix, by applying it to a basis."""
    eye = torch.eye(columns, dtype=like.dtype, device=like.device)
    applied = torch.stack([term.operator(row) for row in eye])  # type: ignore[misc]
    return applied.transpose(-2, -1).expand(*like.shape[:-2], -1, columns)
