r"""Conjugate gradient for the regularised normal equations.

Solves, for the general least-squares problem

.. math::

    \min_x \; \|A x - y\|^2 + \sum_k \lambda_k \|R_k x - c_k\|^2

the normal equations that stationarity gives,

.. math::

    \Big(A^H A + \sum_k \lambda_k R_k^H R_k\Big) x
        = A^H y + \sum_k \lambda_k R_k^H c_k,

by conjugate gradient, optionally preconditioned. The caller supplies
:math:`A^H A` and :math:`A^H y` rather than :math:`A`, because for a
non-Cartesian acquisition the normal operator is a convolution that costs far
less than a transform pair, and because that is the object a Toeplitz
factorisation stands for.

Two properties this implementation is careful about, both learned the hard way.

**It steps through negative curvature.** A compressed Toeplitz normal carries
eigenvalues just below zero -- the transfer values its support left out -- and
the iteration keeps reducing the residual through them. Refusing the step,
which is what a ``pAp > 0`` guard does, freezes the iteration well short of
the answer. BART's ``conjgrad`` stops only on an exactly zero curvature, and
so does this.

**It allocates nothing per iteration that it can avoid.** The updates are
in place and the inner products are fused reductions: ``torch.vdot`` and a
batched ``einsum`` take a dot without materialising the product, where
``(a.conj() * b).real.sum()`` costs two whole volumes and
``torch.linalg.vecdot`` costs the same. What is left is whatever the operator
itself returns.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch.autograd.function import once_differentiable

__all__ = ["CGResult", "Regularizer", "conjugate_gradient"]

Operator = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Regularizer:
    r"""One term :math:`\lambda \|R x - c\|^2` of the objective.

    Parameters
    ----------
    weight
        The term's :math:`\lambda`. Must not be negative.
    operator
        :math:`R`. ``None`` means the identity, which is the common case and
        is folded into a single scalar rather than applied.
    adjoint
        :math:`R^H`. Needed when ``operator`` is given and does not carry its
        own ``adjoint`` or ``H`` attribute.
    bias
        :math:`c`, the term's target. ``None`` means zero, so the term pulls
        towards the origin.

    Examples
    --------
    Pull towards zero, which is Tikhonov regularisation:

    >>> from torchsolve import Regularizer
    >>> Regularizer(1e-3).operator is None
    True

    Pull towards a previous estimate, which is what an outer iteration wants:

    >>> import torch
    >>> previous = torch.zeros(4)
    >>> term = Regularizer(1e-2, bias=previous)
    >>> term.weight
    0.01
    """

    weight: float
    operator: Operator | None = None
    adjoint: Operator | None = None
    bias: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(
                f"regularizer weight must not be negative, got {self.weight}"
            )
        if self.operator is None:
            if self.adjoint is not None:
                raise ValueError("an identity regularizer takes no adjoint")
            return
        if self.adjoint is not None:
            return
        for name in ("adjoint", "H"):
            found = getattr(self.operator, name, None)
            if callable(found):
                object.__setattr__(self, "adjoint", found)
                return
        raise ValueError(
            "a regularizer with an operator needs its adjoint: pass adjoint=, "
            "or give the operator an 'adjoint' or 'H' attribute"
        )

    @property
    def is_identity(self) -> bool:
        """Whether this term regularises towards a target without transforming."""
        return self.operator is None

    def normal(self, vector: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`R^H R` to a vector."""
        forward = cast("Operator", self.operator)
        backward = cast("Operator", self.adjoint)
        return backward(forward(vector))

    def project(self, vector: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`R^H` to a vector, which is what a bias needs."""
        return cast("Operator", self.adjoint)(vector)


@dataclass
class CGResult:
    """What the iteration reached, and what it met on the way.

    Parameters
    ----------
    solution
        The final iterate.
    iterations
        How many steps were taken.
    residual_norm
        Norm of the final residual of the regularised normal equations.
    converged
        Whether the residual met the requested tolerance. Always ``False``
        when no tolerance was requested, since nothing was checked.
    definite
        Whether the recurrence stayed positive throughout. ``False`` means the
        operator is not positive definite and the answer is not a minimiser,
        though it is still the best the iteration reached.
    """

    solution: torch.Tensor
    iterations: int
    residual_norm: torch.Tensor
    converged: bool
    definite: bool = True


@dataclass
class _System:
    """The regularised normal operator, and the right-hand side it acts on."""

    normal: Operator
    tikhonov: float
    terms: Sequence[Regularizer]
    rhs: torch.Tensor = field(repr=False)

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        result = self.normal(vector)
        if self.tikhonov:
            result = result.add(vector, alpha=self.tikhonov)
        for term in self.terms:
            result = result.add(term.normal(vector), alpha=term.weight)
        return result


def _assemble(
    normal: Operator,
    rhs: torch.Tensor,
    regularizers: Iterable[Regularizer],
) -> _System:
    """Fold the identity terms into a scalar and add every bias to the rhs."""
    tikhonov = 0.0
    shaped: list[Regularizer] = []
    augmented = rhs
    for term in regularizers:
        if term.weight == 0:
            continue
        if term.is_identity:
            tikhonov += term.weight
            if term.bias is not None:
                augmented = augmented.add(term.bias, alpha=term.weight)
            continue
        shaped.append(term)
        if term.bias is not None:
            augmented = augmented.add(term.project(term.bias), alpha=term.weight)
    return _System(normal, tikhonov, tuple(shaped), augmented)


def _inner(left: torch.Tensor, right: torch.Tensor, batch: int | None) -> torch.Tensor:
    """Real part of the inner product, without materialising the product.

    ``torch.vdot`` is a fused dot and ``einsum`` reduces without a temporary;
    the obvious ``(left.conj() * right).real.sum()`` costs two whole tensors,
    and so does ``torch.linalg.vecdot``.
    """
    if batch is None:
        flat_left, flat_right = left.reshape(-1), right.reshape(-1)
        if left.is_complex():
            return torch.vdot(flat_left, flat_right).real
        return torch.dot(flat_left, flat_right)
    moved_left = left.movedim(batch, 0).reshape(left.shape[batch], -1)
    moved_right = right.movedim(batch, 0).reshape(right.shape[batch], -1)
    product = torch.einsum("bi,bi->b", moved_left.conj(), moved_right)
    shape = [1] * left.ndim
    shape[batch] = left.shape[batch]
    return (product.real if left.is_complex() else product).reshape(shape)


def _norm(value: torch.Tensor, batch: int | None) -> torch.Tensor:
    """Euclidean norm, fused."""
    if batch is None:
        return torch.linalg.vector_norm(value)
    moved = value.movedim(batch, 0).reshape(value.shape[batch], -1)
    shape = [1] * value.ndim
    shape[batch] = value.shape[batch]
    return torch.linalg.vector_norm(moved, dim=1).reshape(shape)


def _iterate(
    system: _System,
    x0: torch.Tensor | None,
    *,
    preconditioner: Operator | None,
    max_iter: int,
    rtol: float,
    atol: float,
    batch_dim: int | None,
) -> CGResult:
    """Run the recurrence. See :func:`conjugate_gradient` for the arguments."""
    target = system.rhs

    if x0 is None:
        solution = torch.zeros_like(target)
        residual = target.clone()
    else:
        solution = x0.clone()
        residual = target - system(solution)

    preconditioned = residual if preconditioner is None else preconditioner(residual)
    direction = preconditioned.clone()
    rho = _inner(residual, preconditioned, batch_dim)

    checking = rtol > 0.0 or atol > 0.0
    threshold = atol + rtol * _norm(target, batch_dim) if checking else None

    definite = True
    converged = False
    iterations = 0
    for step_index in range(max_iter):
        if threshold is not None and bool(
            torch.all(_norm(residual, batch_dim) <= threshold)
        ):
            converged = True
            break

        # An exactly zero rho is exact convergence, not a failure: there is no
        # residual left to reduce and every further step would be a no-op.
        remaining = rho != 0
        if not bool(torch.any(remaining)):
            converged = True
            break

        applied = system(direction)
        curvature = _inner(direction, applied, batch_dim)
        # Only an exactly zero curvature has no step to take. A negative one
        # does, and taking it is what keeps the residual falling.
        usable = remaining & (curvature != 0)
        if not bool(torch.any(usable)):
            break
        # Negative curvature is what indefiniteness looks like. A vanishing rho
        # is convergence and must not be mistaken for it.
        if definite and bool(torch.any(remaining & (curvature < 0))):
            definite = False

        alpha = torch.where(usable, rho / torch.where(usable, curvature, 1.0), 0.0)
        solution.addcmul_(direction, alpha)
        residual.addcmul_(applied, alpha, value=-1)
        del applied

        preconditioned = (
            residual if preconditioner is None else preconditioner(residual)
        )
        updated = _inner(residual, preconditioned, batch_dim)
        beta = torch.where(rho != 0, updated / torch.where(rho != 0, rho, 1.0), 0.0)
        rho = updated
        iterations = step_index + 1

        if step_index + 1 < max_iter:
            direction.mul_(beta).add_(preconditioned)

    return CGResult(
        solution=solution,
        iterations=iterations,
        residual_norm=_norm(residual, batch_dim).detach(),
        converged=converged,
        definite=definite,
    )


class _ImplicitSolve(torch.autograd.Function):
    """A solve that differentiates itself rather than its iterations.

    The iterates are not a graph to walk back through: for a self-adjoint
    system, if ``x`` solves ``M x = b`` then a gradient arriving at ``x``
    reaches ``b`` as ``M^-1`` of itself, which is another solve. Memory is
    therefore flat in the iteration count, and the backward pass costs one more
    solve rather than one stored volume per step.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        target: torch.Tensor,
        x0: torch.Tensor | None,
        settings: _Settings,
        record: dict[str, Any],
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        system = _System(settings.normal, settings.tikhonov, settings.terms, target)
        with torch.no_grad():
            result = _iterate(
                system,
                x0,
                preconditioner=settings.preconditioner,
                max_iter=settings.max_iter,
                rtol=settings.rtol,
                atol=settings.atol,
                batch_dim=settings.batch_dim,
            )
        record.update(
            iterations=result.iterations,
            residual_norm=result.residual_norm,
            converged=result.converged,
            definite=result.definite,
        )
        ctx.settings = settings
        ctx.save_for_backward(result.solution, *parameters)
        return result.solution

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_solution: torch.Tensor) -> tuple[Any, ...]:
        solution, *parameters = ctx.saved_tensors
        settings: _Settings = ctx.settings
        system = _System(
            settings.normal, settings.tikhonov, settings.terms, grad_solution
        )
        with torch.no_grad():
            adjoint = _iterate(
                system,
                None,
                preconditioner=settings.preconditioner,
                max_iter=settings.backward_max_iter,
                rtol=settings.backward_rtol,
                atol=settings.backward_atol,
                batch_dim=settings.batch_dim,
            ).solution

        gradients: list[torch.Tensor | None] = [None] * len(parameters)
        wanted = [
            (index, parameter)
            for index, parameter in enumerate(parameters)
            if parameter.requires_grad
        ]
        if wanted:
            # d/dp of (M(p) x - b) = 0 gives -adjoint^H (dM/dp) x, which is the
            # gradient of this scalar with respect to whatever M closed over.
            with torch.enable_grad():
                applied = system.normal(solution.detach())
                for term in system.terms:
                    applied = applied.add(
                        term.normal(solution.detach()), alpha=term.weight
                    )
                pseudo = -_inner(adjoint.detach(), applied, None)
            if pseudo.requires_grad:
                found = torch.autograd.grad(
                    pseudo,
                    [parameter for _, parameter in wanted],
                    allow_unused=True,
                )
                for (index, _), gradient in zip(wanted, found, strict=True):
                    gradients[index] = gradient
        return adjoint, None, None, None, *gradients


@dataclass(frozen=True)
class _Settings:
    """Everything the backward pass has to reconstruct the system from."""

    normal: Operator
    tikhonov: float
    terms: Sequence[Regularizer]
    preconditioner: Operator | None
    max_iter: int
    rtol: float
    atol: float
    backward_max_iter: int
    backward_rtol: float
    backward_atol: float
    batch_dim: int | None


def conjugate_gradient(
    normal: Operator,
    rhs: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    regularizers: Iterable[Regularizer] = (),
    preconditioner: Operator | None = None,
    max_iter: int = 10,
    rtol: float = 0.0,
    atol: float = 0.0,
    batch_dim: int | None = None,
    parameters: Iterable[torch.Tensor] = (),
    backward_max_iter: int | None = None,
    backward_rtol: float | None = None,
    backward_atol: float | None = None,
    warn_indefinite: bool = True,
) -> CGResult:
    r"""Solve the regularised normal equations by conjugate gradient.

    Parameters
    ----------
    normal
        :math:`A^H A`, as a callable taking and returning one tensor.
    rhs
        :math:`A^H y`.
    x0
        Starting iterate. ``None`` starts from zero, which also saves the first
        operator application, because the residual is then the right-hand side.
    regularizers
        The :class:`Regularizer` terms. Identity terms are summed into one
        scalar and applied without an operator call.
    preconditioner
        :math:`M^{-1}`, applied to the residual each iteration. This is what
        replaces density compensation for a non-Cartesian acquisition: the
        weighting belongs in the solver, where it changes only the path taken,
        rather than in the data, where it changes the answer.
    max_iter
        Iteration cap.
    rtol, atol
        Stop once ``||r|| <= atol + rtol * ||b||``. Both zero, the default,
        runs the full iteration count and never reads a value back to the host,
        so the loop does not synchronise.
    batch_dim
        Axis along which the problem is a batch of independent systems, each
        with its own step size. ``None`` treats the whole tensor as one system.
    parameters
        Tensors inside ``normal`` or the regularizers that gradients are wanted
        for. A closure cannot be inspected, so they are named here.
    backward_max_iter, backward_rtol, backward_atol
        Settings for the solve the backward pass runs. Each defaults to its
        forward counterpart.
    warn_indefinite
        Whether to warn when the recurrence meets negative curvature.

    Returns
    -------
    CGResult
        The iterate, and what the iteration met on the way.

    Notes
    -----
    Differentiable in ``rhs`` and in ``parameters``, by implicit
    differentiation rather than by unrolling: memory is flat in ``max_iter``
    and the backward pass is one more solve.

    Examples
    --------
    >>> import torch
    >>> from torchsolve import Regularizer, conjugate_gradient
    >>> matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    >>> truth = torch.tensor([1.0, 2.0])
    >>> result = conjugate_gradient(lambda v: matrix @ v, matrix @ truth, max_iter=8)
    >>> bool(torch.allclose(result.solution, truth, atol=1e-5))
    True

    Regularising towards zero shrinks the answer:

    >>> pulled = conjugate_gradient(
    ...     lambda v: matrix @ v,
    ...     matrix @ truth,
    ...     regularizers=[Regularizer(100.0)],
    ...     max_iter=8,
    ... )
    >>> bool(pulled.solution.norm() < result.solution.norm())
    True

    A gradient reaches the right-hand side through the solve:

    >>> data = (matrix @ truth).requires_grad_(True)
    >>> conjugate_gradient(lambda v: matrix @ v, data, max_iter=8).solution.sum().backward()
    >>> bool(data.grad.abs().sum() > 0)
    True
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    if rtol < 0 or atol < 0:
        raise ValueError("tolerances must not be negative")

    system = _assemble(normal, rhs, regularizers)
    held = tuple(parameters)
    differentiating = torch.is_grad_enabled() and (
        system.rhs.requires_grad or any(one.requires_grad for one in held)
    )

    if not differentiating:
        result = _iterate(
            system,
            x0,
            preconditioner=preconditioner,
            max_iter=max_iter,
            rtol=rtol,
            atol=atol,
            batch_dim=batch_dim,
        )
    else:
        settings = _Settings(
            normal=system.normal,
            tikhonov=system.tikhonov,
            terms=system.terms,
            preconditioner=preconditioner,
            max_iter=max_iter,
            rtol=rtol,
            atol=atol,
            backward_max_iter=max_iter
            if backward_max_iter is None
            else backward_max_iter,
            backward_rtol=rtol if backward_rtol is None else backward_rtol,
            backward_atol=atol if backward_atol is None else backward_atol,
            batch_dim=batch_dim,
        )
        record: dict[str, Any] = {}
        solution = _ImplicitSolve.apply(system.rhs, x0, settings, record, *held)
        result = CGResult(solution=solution, **record)

    if not result.definite and warn_indefinite:
        warnings.warn(
            "conjugate gradient met negative curvature: the operator is not "
            "positive definite, so the answer is not a minimiser. It is still "
            "the best iterate reached. Raise the regularisation, stop earlier, "
            "or keep the whole transfer if it is a compressed one.",
            stacklevel=2,
        )
    return result
