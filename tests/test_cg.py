"""Conjugate gradient on the regularised normal equations."""

from __future__ import annotations

import pytest
import torch

from torchsolve import Regularizer, conjugate_gradient

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def spd_matrix(size: int, seed: int = 0, spread: float = 1.0) -> torch.Tensor:
    """A symmetric positive definite matrix with a controllable spread."""
    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(size, size, generator=generator))[0]
    eigenvalues = torch.logspace(0, spread, size, dtype=torch.float64)
    return (basis.double() * eigenvalues) @ basis.double().T


def bart_conjgrad(operator, target, start, l2lambda, max_iter, epsilon):
    """A transcription of BART's ``conjgrad`` from ``src/iter/italgos.c``.

    Kept literal, out-of-place and unoptimised, so that any disagreement is a
    disagreement about the algorithm rather than about how it was written.
    """
    solution = start.clone()
    residual = operator(solution) + l2lambda * solution
    residual = target - residual
    direction = residual.clone()
    rsold = residual.norm() ** 2
    rsnew = rsold
    eps_squared = epsilon**2
    if rsold == 0:
        return solution
    for _ in range(max_iter):
        applied = operator(direction) + l2lambda * direction
        curvature = torch.dot(direction, applied)
        if curvature == 0:
            break
        alpha = rsold / curvature
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        rsnew = residual.norm() ** 2
        beta = rsnew / rsold
        rsold = rsnew
        if rsnew <= eps_squared:
            break
        direction = beta * direction + residual
    return solution


def test_matches_bart_while_the_recurrence_is_well_conditioned() -> None:
    """The recurrence is BART's, to the precision the arithmetic allows.

    The two agree to machine epsilon until CG's Krylov basis starts losing
    orthogonality, after which any rearrangement of the same arithmetic
    diverges: the in-place fused updates here round differently from the
    out-of-place ones BART writes, and the iteration amplifies that. What is
    asserted is the algorithm, not the rounding -- and then, separately, that
    both arrive at the same answer.
    """
    matrix = spd_matrix(32, spread=2.0)
    truth = torch.randn(32, generator=torch.Generator().manual_seed(1)).double()
    target = matrix @ truth

    def operator(vector):
        return matrix @ vector

    for iterations in (1, 2, 3, 5, 7, 10):
        reference = bart_conjgrad(
            operator, target, torch.zeros(32).double(), 0.0, iterations, 0.0
        )
        ours = conjugate_gradient(
            operator, target, max_iter=iterations, x0=torch.zeros(32).double()
        ).solution
        assert torch.allclose(ours, reference, rtol=1e-11, atol=1e-12), iterations


def test_reaches_the_same_answer_as_bart() -> None:
    """Run both to convergence and they land on the solution together."""
    matrix = spd_matrix(32, spread=2.0)
    truth = torch.randn(32, generator=torch.Generator().manual_seed(1)).double()
    target = matrix @ truth

    def operator(vector):
        return matrix @ vector

    reference = bart_conjgrad(operator, target, torch.zeros(32).double(), 0.0, 400, 0.0)
    ours = conjugate_gradient(
        operator, target, max_iter=400, x0=torch.zeros(32).double(), rtol=1e-13
    ).solution
    assert torch.allclose(ours, truth, atol=1e-9)
    assert torch.allclose(reference, truth, atol=1e-9)


def test_matches_bart_with_the_tikhonov_term() -> None:
    """BART's ``l2lambda`` is an identity regularizer pulling towards zero."""
    matrix = spd_matrix(24, spread=2.0)
    truth = torch.randn(24, generator=torch.Generator().manual_seed(2)).double()
    target = matrix @ truth

    def operator(vector):
        return matrix @ vector

    reference = bart_conjgrad(operator, target, torch.zeros(24).double(), 0.05, 12, 0.0)
    ours = conjugate_gradient(
        operator,
        target,
        regularizers=[Regularizer(0.05)],
        max_iter=12,
        x0=torch.zeros(24).double(),
    ).solution
    assert torch.allclose(ours, reference, rtol=1e-9, atol=1e-11)


def test_solves_a_well_posed_system() -> None:
    matrix = spd_matrix(40, spread=1.5)
    truth = torch.randn(40, generator=torch.Generator().manual_seed(3)).double()
    result = conjugate_gradient(
        lambda v: matrix @ v, matrix @ truth, max_iter=200, rtol=1e-12
    )
    assert result.converged
    assert result.definite
    assert torch.allclose(result.solution, truth, atol=1e-8)


def test_starting_from_zero_skips_an_operator_call() -> None:
    matrix = spd_matrix(16)
    calls = []

    def counted(vector):
        calls.append(None)
        return matrix @ vector

    target = matrix @ torch.ones(16).double()
    conjugate_gradient(counted, target, max_iter=4)
    from_zero = len(calls)
    calls.clear()
    conjugate_gradient(counted, target, max_iter=4, x0=torch.zeros(16).double())
    assert len(calls) == from_zero + 1


def test_steps_through_negative_curvature() -> None:
    """An indefinite operator must not stall the iteration.

    A compressed Toeplitz normal carries eigenvalues just below zero. A
    ``curvature > 0`` guard, which is what MIRTorch applies, stops on them and
    leaves the residual where it was.
    """
    size = 24
    matrix = spd_matrix(size, seed=4, spread=1.0)
    eigenvalues, basis = torch.linalg.eigh(matrix)
    eigenvalues[0] = -0.05 * eigenvalues[-1]  # one direction of negative curvature
    matrix = (basis * eigenvalues) @ basis.T
    truth = torch.randn(size, generator=torch.Generator().manual_seed(5)).double()
    target = matrix @ truth

    def operator(vector):
        return matrix @ vector

    with pytest.warns(UserWarning, match="negative curvature"):
        result = conjugate_gradient(operator, target, max_iter=60)
    assert not result.definite
    assert result.iterations == 60
    # The residual fell a long way despite the indefiniteness.
    assert result.residual_norm < 1e-3 * target.norm()

    # A guard on positive curvature would have stopped almost immediately.
    stalled = _guarded_cg(operator, target, max_iter=60)
    assert result.residual_norm < stalled


def _guarded_cg(operator, target, max_iter):
    """CG that refuses negative curvature, as MIRTorch's does."""
    solution = torch.zeros_like(target)
    residual = target.clone()
    direction = residual.clone()
    rho = residual.norm() ** 2
    for _ in range(max_iter):
        applied = operator(direction)
        curvature = torch.dot(direction, applied)
        if not (curvature > 0 and rho > 0):
            break
        alpha = rho / curvature
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        updated = residual.norm() ** 2
        direction = (updated / rho) * direction + residual
        rho = updated
    return residual.norm()


def test_preconditioning_shortens_a_badly_scaled_solve() -> None:
    size = 60
    scale = torch.logspace(0, 3, size, dtype=torch.float64)
    matrix = torch.diag(scale) + 1e-3 * spd_matrix(size, seed=6, spread=0.1)
    truth = torch.randn(size, generator=torch.Generator().manual_seed(7)).double()
    target = matrix @ truth

    def operator(v):
        return matrix @ v

    def jacobi(v):
        return v / scale

    plain = conjugate_gradient(operator, target, max_iter=15)
    shaped = conjugate_gradient(operator, target, max_iter=15, preconditioner=jacobi)
    assert shaped.residual_norm < 0.1 * plain.residual_norm


def test_identity_regularizer_pulls_towards_its_bias() -> None:
    matrix = spd_matrix(12)
    truth = torch.randn(12, generator=torch.Generator().manual_seed(8)).double()
    target = matrix @ truth
    anchor = torch.full((12,), 5.0, dtype=torch.float64)
    pulled = conjugate_gradient(
        lambda v: matrix @ v,
        target,
        regularizers=[Regularizer(1e4, bias=anchor)],
        max_iter=40,
    ).solution
    assert torch.allclose(pulled, anchor, atol=1e-2)


def test_shaped_regularizer_uses_its_adjoint() -> None:
    """A first-difference penalty smooths the answer."""
    size = 33
    matrix = torch.eye(size, dtype=torch.float64)
    rough = torch.zeros(size, dtype=torch.float64)
    rough[::2] = 1.0
    difference = torch.diff(torch.eye(size, dtype=torch.float64), dim=0)

    smoothed = conjugate_gradient(
        lambda v: matrix @ v,
        rough,
        regularizers=[
            Regularizer(
                3.0,
                operator=lambda v: difference @ v,
                adjoint=lambda v: difference.T @ v,
            )
        ],
        max_iter=200,
        rtol=1e-10,
    ).solution
    assert torch.diff(smoothed).abs().sum() < 0.5 * torch.diff(rough).abs().sum()


def test_adjoint_is_found_on_the_operator() -> None:
    class Difference:
        def __init__(self, size):
            self.matrix = torch.diff(torch.eye(size, dtype=torch.float64), dim=0)

        def __call__(self, vector):
            return self.matrix @ vector

        def adjoint(self, vector):
            return self.matrix.T @ vector

    term = Regularizer(1.0, operator=Difference(8))
    assert term.adjoint is not None


def test_operator_without_an_adjoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs its adjoint"):
        Regularizer(1.0, operator=lambda v: v)


def test_identity_regularizer_rejects_an_adjoint() -> None:
    with pytest.raises(ValueError, match="no adjoint"):
        Regularizer(1.0, adjoint=lambda v: v)


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        Regularizer(-1.0)


def test_batched_systems_each_take_their_own_step() -> None:
    """Two systems of very different scale, solved together."""
    size = 20
    first = spd_matrix(size, seed=9, spread=0.5)
    second = 1000.0 * spd_matrix(size, seed=10, spread=0.5)
    stacked = torch.stack([first, second])
    truth = torch.randn(2, size, generator=torch.Generator().manual_seed(11)).double()
    target = torch.einsum("bij,bj->bi", stacked, truth)

    def operator(v):
        return torch.einsum("bij,bj->bi", stacked, v)

    batched = conjugate_gradient(operator, target, max_iter=120, batch_dim=0)
    assert torch.allclose(batched.solution, truth, atol=1e-6)

    # Treated as one system, the shared step size serves neither well.
    together = conjugate_gradient(operator, target, max_iter=120)
    assert (batched.solution - truth).norm() < (together.solution - truth).norm()


def test_complex_system() -> None:
    size = 16
    generator = torch.Generator().manual_seed(12)
    root = torch.randn(size, size, generator=generator, dtype=torch.complex128)
    matrix = root.conj().T @ root + size * torch.eye(size, dtype=torch.complex128)
    truth = torch.randn(size, generator=generator, dtype=torch.complex128)
    result = conjugate_gradient(
        lambda v: matrix @ v, matrix @ truth, max_iter=200, rtol=1e-12
    )
    assert torch.allclose(result.solution, truth, atol=1e-8)


def test_zero_weight_regularizer_changes_nothing() -> None:
    matrix = spd_matrix(10)
    target = matrix @ torch.ones(10).double()
    plain = conjugate_gradient(lambda v: matrix @ v, target, max_iter=6).solution
    padded = conjugate_gradient(
        lambda v: matrix @ v,
        target,
        regularizers=[
            Regularizer(0.0, bias=torch.full((10,), 9.0, dtype=torch.float64))
        ],
        max_iter=6,
    ).solution
    assert torch.equal(plain, padded)


def test_max_iter_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        conjugate_gradient(lambda v: v, torch.ones(3), max_iter=0)


def test_gradient_reaches_the_right_hand_side() -> None:
    """Implicit differentiation, checked against differentiating the inverse."""
    size = 8
    matrix = spd_matrix(size, seed=13)
    data = torch.randn(size, generator=torch.Generator().manual_seed(14)).double()

    through_solve = data.clone().requires_grad_(True)
    conjugate_gradient(
        lambda v: matrix @ v, through_solve, max_iter=200, rtol=1e-14
    ).solution.pow(2).sum().backward()

    through_inverse = data.clone().requires_grad_(True)
    torch.linalg.solve(matrix, through_inverse).pow(2).sum().backward()

    assert torch.allclose(through_solve.grad, through_inverse.grad, atol=1e-7)


def test_gradient_reaches_an_operator_parameter() -> None:
    """A weight inside the operator gets its gradient from the pseudo-loss."""
    size = 6
    basis = spd_matrix(size, seed=15)

    def loss_at(value, differentiable):
        weight = torch.tensor(value, dtype=torch.float64, requires_grad=differentiable)
        target = torch.ones(size, dtype=torch.float64)
        solved = conjugate_gradient(
            lambda v: basis @ v + weight * v,
            target,
            max_iter=300,
            rtol=1e-14,
            parameters=[weight],
        ).solution
        return solved.pow(2).sum(), weight

    loss, weight = loss_at(0.7, True)
    loss.backward()
    step = 1e-5
    upper, _ = loss_at(0.7 + step, False)
    lower, _ = loss_at(0.7 - step, False)
    numerical = (upper - lower) / (2 * step)
    assert weight.grad is not None
    assert abs(weight.grad.item() - numerical.item()) < 1e-4 * abs(numerical.item())


def test_gradient_memory_does_not_grow_with_iterations() -> None:
    """Nothing per-iteration is stored, so the graph is the same size either way."""
    size = 64
    matrix = spd_matrix(size, seed=16)
    data = torch.ones(size, dtype=torch.float64, requires_grad=True)

    def graph_tensors(iterations):
        result = conjugate_gradient(
            lambda v: matrix @ v, data, max_iter=iterations
        ).solution
        node = result.grad_fn
        seen = 0
        stack = [node]
        while stack:
            current = stack.pop()
            if current is None:
                continue
            seen += 1
            stack.extend(parent for parent, _ in current.next_functions)
        return seen

    assert graph_tensors(4) == graph_tensors(64)


@requires_cuda
def test_iteration_allocates_only_what_the_operator_returns() -> None:
    """Peak memory must not grow with the iteration count."""
    size = 96
    volume = torch.randn(size, size, size, dtype=torch.complex64, device="cuda")
    scale = 2.0 + torch.rand_like(volume.real)

    def normal(vector):
        return scale * vector

    def peak(iterations):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        conjugate_gradient(normal, volume, max_iter=iterations)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - before

    few, many = peak(3), peak(50)
    one_volume = volume.numel() * 8
    assert many <= few + 0.25 * one_volume, (few / one_volume, many / one_volume)
    # solution, residual, direction and the operator's return: four volumes,
    # and the preconditioned residual aliases the residual when there is none.
    assert many <= 5 * one_volume
