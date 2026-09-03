r"""Iteratively regularised Gauss-Newton (Bakushinsky 1993).

For a nonlinear :math:`F` and data :math:`y`, linearise about the current
estimate and solve the regularised linear problem the linearisation gives:

.. math::

    \big(DF^H DF + \alpha\big)\,(x - x_{\text{ref}})
        = DF^H\big(y - F(x) + DF\,(x - x_{\text{ref}})\big),

then decrease :math:`\alpha` and repeat. The regularisation starts strong,
which is what keeps the first steps from chasing a linearisation that is only
locally true, and is relaxed geometrically as the estimate improves. That
schedule is the method: a Gauss-Newton step with a *fixed* regularisation is
something else and behaves differently.

This follows BART's ``irgnm2`` in ``src/iter/italgos.c``, which solves for
:math:`x - x_{\text{ref}}` rather than for the update, at the cost of one extra
derivative call, so that the inner solve is an ordinary regularised
least-squares problem and any solver can do it.

**Constraints do not need a solver that knows about them.** A non-negative
parameter is one written as :math:`x = e^{\theta}` or :math:`\theta^2`, and an
equality like :math:`w + f = 1` is one free parameter fewer -- write
:math:`F(f)` with :math:`w` replaced by :math:`1 - f`. Both are changes of
variable, so both are exact, cost nothing, and reach the solver as an ordinary
unconstrained problem in :math:`\theta`. What they do change is the geometry
the Gauss-Newton step is taken in, which is usually a help and occasionally a
hindrance: :math:`\theta^2` has a vanishing derivative at zero, so an estimate
driven to the bound stops moving. What genuinely needs more than this is a
non-smooth penalty or a constraint coupling many parameters at once, and that
is what the solver interface is for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from ._cg import Regularizer
from ._problem import LinearProblem
from ._solvers import CGSolver

__all__ = ["GaussNewtonResult", "NonlinearOperator", "autodiff", "gauss_newton"]


@runtime_checkable
class NonlinearOperator(Protocol):
    """A model, its derivative, and the derivative's adjoint.

    Everything is evaluated at an explicit point, so nothing is cached between
    calls and a step can be retried without re-establishing state.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate :math:`F(x)`."""
        ...

    def derivative(self, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Apply :math:`DF(x)` to a step."""
        ...

    def adjoint(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply :math:`DF(x)^H` to a data-space vector."""
        ...


@dataclass
class autodiff:
    """Give a plain callable the derivatives, by automatic differentiation.

    Parameters
    ----------
    forward
        :math:`F`, any differentiable callable.
    batch_dims
        How many leading axes of the argument are a batch of independent
        problems. Only :meth:`jacobian` needs to know.

    Examples
    --------
    >>> import torch
    >>> from torchsolve import autodiff
    >>> model = autodiff(lambda x: x**2)
    >>> point = torch.tensor([3.0])
    >>> model(point)
    tensor([9.])
    >>> model.derivative(point, torch.tensor([1.0]))
    tensor([6.])
    """

    forward: Callable[[torch.Tensor], torch.Tensor]
    batch_dims: int = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the model."""
        return self.forward(x)

    def derivative(self, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Apply the derivative, by a forward-mode product."""
        return torch.func.jvp(self.forward, (x,), (step,))[1]

    def adjoint(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint, by a reverse-mode product."""
        _, pullback = torch.func.vjp(self.forward, x)
        return pullback(residual)[0]

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Return the dense Jacobian, for a direct solve of a small problem."""
        compute = torch.func.jacrev(self.forward)
        for _ in range(self.batch_dims):
            compute = torch.func.vmap(compute)
        return compute(x)


@dataclass
class GaussNewtonResult:
    """The estimate, and how it got there.

    Parameters
    ----------
    solution
        The final estimate.
    residual_norms
        Norm of ``y - F(x)`` before each step, so its length is the number of
        steps taken and its trend is what says whether the schedule worked.
    alphas
        The regularisation used at each step.
    """

    solution: torch.Tensor
    residual_norms: list[float] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)


def _as_operator(operator: Any, batch_dims: int) -> NonlinearOperator:
    """Accept a full operator, or make one out of a plain callable."""
    if hasattr(operator, "derivative") and hasattr(operator, "adjoint"):
        return operator
    return autodiff(operator, batch_dims=batch_dims)


def gauss_newton(
    operator: Any,
    data: torch.Tensor,
    x0: torch.Tensor,
    *,
    solver: Callable[[LinearProblem], torch.Tensor] | None = None,
    iterations: int = 8,
    alpha: float = 1.0,
    alpha_min: float = 0.0,
    reduction: float = 2.0,
    reference: torch.Tensor | None = None,
    regularizers: Iterable[Regularizer] = (),
    batch_dims: int = 0,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> GaussNewtonResult:
    r"""Fit a nonlinear model by iteratively regularised Gauss-Newton.

    Parameters
    ----------
    operator
        The model. Either something satisfying :class:`NonlinearOperator`, or
        any differentiable callable, which is wrapped in :class:`autodiff`.
    data
        :math:`y`.
    x0
        The starting estimate, and the reference the regularisation pulls
        towards unless ``reference`` says otherwise.
    solver
        Anything that solves a :class:`LinearProblem`. Defaults to
        :class:`~torchsolve.CGSolver`. This is the seam: a proximal or ADMM
        solver belongs here rather than in this function.
    iterations
        Newton steps.
    alpha
        Starting regularisation. Every regularizer is scaled by it.
    alpha_min
        Floor the schedule decays towards.
    reduction
        Divisor per step: ``alpha <- (alpha - alpha_min) / reduction +
        alpha_min``. BART's default of 2 halves it each step.
    reference
        :math:`x_{\text{ref}}`. Defaults to ``x0``.
    regularizers
        Terms whose weights are *relative* to ``alpha``. The default is a
        single identity term at weight one: Tikhonov towards the reference.
    batch_dims
        Leading axes of ``x0`` that are a batch of independent problems.
    callback
        Called with the step index and the current estimate.

    Returns
    -------
    GaussNewtonResult
        The estimate and the two histories worth keeping.

    Examples
    --------
    Two-parameter exponential decay, fitted from a noiseless curve:

    >>> import torch
    >>> from torchsolve import gauss_newton
    >>> time = torch.linspace(0, 2, 40, dtype=torch.float64)
    >>> def model(p):
    ...     return p[0] * torch.exp(-p[1] * time)
    >>> truth = torch.tensor([2.0, 1.5], dtype=torch.float64)
    >>> start = torch.tensor([1.0, 1.0], dtype=torch.float64)
    >>> found = gauss_newton(model, model(truth), start, iterations=12, alpha=1e-2)
    >>> bool(torch.allclose(found.solution, truth, atol=1e-4))
    True
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    if reduction <= 0:
        raise ValueError(f"reduction must be positive, got {reduction}")
    if alpha < 0 or alpha_min < 0:
        raise ValueError("regularisation must not be negative")

    model = _as_operator(operator, batch_dims)
    inner = CGSolver() if solver is None else solver
    anchor = x0 if reference is None else reference
    estimate = x0.clone()
    terms = tuple(regularizers)
    result = GaussNewtonResult(solution=estimate)
    current = alpha

    for step in range(iterations):
        point = estimate
        residual = data - model(point)
        result.residual_norms.append(float(torch.linalg.vector_norm(residual)))
        result.alphas.append(current)

        # BART linearises at the current estimate but applies the derivative to
        # its offset from the reference, which is what makes the solve return
        # x - xref rather than the update.
        offset = point - anchor
        residual = residual + model.derivative(point, offset)
        rhs = model.adjoint(point, residual)

        def normal(vector: torch.Tensor, _at: torch.Tensor = point) -> torch.Tensor:
            return model.adjoint(_at, model.derivative(_at, vector))

        problem = LinearProblem(
            normal=normal,
            rhs=rhs,
            alpha=current,
            regularizers=terms,
            x0=offset,
        )
        jacobian = getattr(model, "jacobian", None)
        if jacobian is not None and _wants_matrix(inner):
            problem.matrix = jacobian(point)
            problem.target = residual

        estimate = anchor + inner(problem)
        current = max((current - alpha_min) / reduction + alpha_min, alpha_min)
        if callback is not None:
            callback(step, estimate)

    result.solution = estimate
    return result


def _wants_matrix(solver: Any) -> bool:
    """Whether this solver factorises rather than iterates."""
    return getattr(solver, "needs_matrix", False)
