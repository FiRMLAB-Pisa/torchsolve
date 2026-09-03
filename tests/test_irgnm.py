"""Iteratively regularised Gauss-Newton, and the solvers it delegates to."""

from __future__ import annotations

import pytest
import torch

from torchsolve import (
    CGSolver,
    LinearProblem,
    LstsqSolver,
    Regularizer,
    autodiff,
    gauss_newton,
)

TIME = torch.linspace(0.0, 2.0, 48, dtype=torch.float64)


def decay(parameters: torch.Tensor) -> torch.Tensor:
    """Amplitude and rate, as a mono-exponential."""
    return parameters[..., :1] * torch.exp(-parameters[..., 1:2] * TIME)


def bart_irgnm2(model, data, x0, iterations, alpha, alpha_min, reduction):
    """A transcription of BART's ``irgnm2`` from ``src/iter/italgos.c``.

    Written out step for step, with the inner least-squares solved densely so
    that only the outer recurrence is under test.
    """
    estimate = x0.clone()
    reference = x0.clone()
    size = x0.numel()
    for _ in range(iterations):
        residual = data - model(estimate)
        estimate = estimate - reference
        jacobian = torch.func.jacrev(model)(reference + estimate)
        residual = residual + jacobian @ estimate
        rhs = jacobian.conj().T @ residual
        normal = jacobian.conj().T @ jacobian + alpha * torch.eye(size, dtype=x0.dtype)
        estimate = torch.linalg.solve(normal, rhs)
        estimate = estimate + reference
        alpha = (alpha - alpha_min) / reduction + alpha_min
    return estimate


def test_matches_bart_irgnm2() -> None:
    truth = torch.tensor([2.0, 1.5], dtype=torch.float64)
    start = torch.tensor([1.0, 0.8], dtype=torch.float64)
    data = decay(truth)

    exact = LstsqSolver()
    ours = gauss_newton(
        decay, data, start, solver=exact, iterations=6, alpha=0.5, reduction=2.0
    ).solution
    reference = bart_irgnm2(decay, data, start, 6, 0.5, 0.0, 2.0)
    assert torch.allclose(ours, reference, rtol=1e-8, atol=1e-10)


def test_recovers_a_mono_exponential() -> None:
    truth = torch.tensor([2.0, 1.5], dtype=torch.float64)
    start = torch.tensor([0.5, 0.5], dtype=torch.float64)
    found = gauss_newton(decay, decay(truth), start, iterations=14, alpha=1e-2)
    assert torch.allclose(found.solution, truth, atol=1e-5)
    # The schedule is the method: the residual falls and alpha halves.
    assert found.residual_norms[-1] < 1e-3 * found.residual_norms[0]
    assert found.alphas[1] == pytest.approx(found.alphas[0] / 2)


def test_regularisation_pulls_towards_the_reference() -> None:
    truth = torch.tensor([2.0, 1.5], dtype=torch.float64)
    start = torch.tensor([0.5, 0.5], dtype=torch.float64)
    held = gauss_newton(
        decay, decay(truth), start, iterations=4, alpha=50.0, reduction=1.0
    ).solution
    free = gauss_newton(decay, decay(truth), start, iterations=4, alpha=1e-3).solution
    assert (held - start).norm() < (free - start).norm()


def test_alpha_floor_is_respected() -> None:
    truth = torch.tensor([2.0, 1.5], dtype=torch.float64)
    start = torch.tensor([1.0, 1.0], dtype=torch.float64)
    found = gauss_newton(
        decay,
        decay(truth),
        start,
        iterations=10,
        alpha=1.0,
        alpha_min=0.25,
        reduction=2.0,
    )
    assert min(found.alphas) >= 0.25


def test_non_negativity_by_change_of_variables() -> None:
    """A bound needs no solver support: write the parameter as an exponential."""
    truth = torch.tensor([2.0, 1.5], dtype=torch.float64)
    data = decay(truth)

    def positive(logs):
        return decay(logs.exp())

    start = torch.zeros(2, dtype=torch.float64)
    found = gauss_newton(positive, data, start, iterations=20, alpha=1e-2)
    recovered = found.solution.exp()
    assert torch.all(recovered > 0)
    assert torch.allclose(recovered, truth, atol=1e-4)


def test_equality_by_eliminating_a_parameter() -> None:
    """w + f = 1 is one parameter fewer, not a constraint to enforce."""
    time = torch.linspace(0.0, 1.0, 40, dtype=torch.float64)

    def two_pool(fraction):
        water = 1.0 - fraction
        return water * torch.exp(-time) + fraction * torch.exp(-4.0 * time)

    truth = torch.tensor([0.3], dtype=torch.float64)
    found = gauss_newton(
        two_pool,
        two_pool(truth),
        torch.tensor([0.6], dtype=torch.float64),
        iterations=12,
        alpha=1e-3,
    ).solution
    assert torch.allclose(found, truth, atol=1e-6)
    # The sum holds by construction, at every value the solver could return.
    assert float((1.0 - found) + found) == pytest.approx(1.0)


def test_lstsq_solver_generalises_torch_lstsq() -> None:
    """With no regularisation it is torch.linalg.lstsq."""
    generator = torch.Generator().manual_seed(0)
    jacobian = torch.randn(30, 4, generator=generator, dtype=torch.float64)
    target = torch.randn(30, generator=generator, dtype=torch.float64)
    problem = LinearProblem(
        normal=lambda v: jacobian.T @ (jacobian @ v),
        rhs=jacobian.T @ target,
        alpha=0.0,
        matrix=jacobian,
        target=target,
    )
    ours = LstsqSolver()(problem)
    reference = torch.linalg.lstsq(jacobian, target.unsqueeze(-1)).solution.squeeze(-1)
    assert torch.allclose(ours, reference, atol=1e-10)


def test_lstsq_solver_matches_the_normal_equations() -> None:
    """Stacking and squaring agree; stacking is the better conditioned of the two."""
    generator = torch.Generator().manual_seed(1)
    jacobian = torch.randn(20, 5, generator=generator, dtype=torch.float64)
    target = torch.randn(20, generator=generator, dtype=torch.float64)
    penalty = torch.diff(torch.eye(5, dtype=torch.float64), dim=0)
    alpha = 0.3
    problem = LinearProblem(
        normal=lambda v: jacobian.T @ (jacobian @ v),
        rhs=jacobian.T @ target,
        alpha=alpha,
        regularizers=[
            Regularizer(
                1.0,
                operator=lambda v: penalty @ v,
                adjoint=lambda v: penalty.T @ v,
            )
        ],
        matrix=jacobian,
        target=target,
    )
    ours = LstsqSolver()(problem)
    squared = torch.linalg.solve(
        jacobian.T @ jacobian + alpha * penalty.T @ penalty,
        jacobian.T @ target,
    )
    assert torch.allclose(ours, squared, atol=1e-9)


def test_one_alpha_governs_every_regularizer() -> None:
    """A scalar alpha scales all the terms, which is what makes it a schedule."""
    generator = torch.Generator().manual_seed(2)
    jacobian = torch.randn(20, 4, generator=generator, dtype=torch.float64)
    target = torch.randn(20, generator=generator, dtype=torch.float64)
    first = torch.diff(torch.eye(4, dtype=torch.float64), dim=0)
    problem = LinearProblem(
        normal=lambda v: jacobian.T @ (jacobian @ v),
        rhs=jacobian.T @ target,
        alpha=0.7,
        regularizers=[
            Regularizer(1.0),
            Regularizer(
                1.0, operator=lambda v: first @ v, adjoint=lambda v: first.T @ v
            ),
        ],
    )
    scaled = problem.scaled()
    assert [term.weight for term in scaled] == [0.7, 0.7]


def test_no_regularizers_means_tikhonov_at_alpha() -> None:
    problem = LinearProblem(normal=lambda v: v, rhs=torch.zeros(3), alpha=0.4)
    only = problem.scaled()
    assert len(only) == 1
    assert only[0].is_identity
    assert only[0].weight == pytest.approx(0.4)


def test_batched_multivoxel_fit() -> None:
    """Every voxel is its own tiny problem, solved in one call."""
    voxels = 64
    generator = torch.Generator().manual_seed(3)
    truth = torch.stack(
        [
            1.0 + torch.rand(voxels, generator=generator, dtype=torch.float64),
            0.5 + 2.0 * torch.rand(voxels, generator=generator, dtype=torch.float64),
        ],
        dim=-1,
    )

    def many(parameters):
        return parameters[..., :1] * torch.exp(-parameters[..., 1:2] * TIME)

    start = torch.stack(
        [
            torch.ones(voxels, dtype=torch.float64),
            torch.ones(voxels, dtype=torch.float64),
        ],
        dim=-1,
    )
    found = gauss_newton(
        many,
        many(truth),
        start,
        solver=LstsqSolver(),
        iterations=16,
        alpha=1e-2,
        batch_dims=1,
    ).solution
    assert found.shape == truth.shape
    assert torch.allclose(found, truth, atol=1e-4)


def test_a_foreign_solver_is_accepted() -> None:
    """The seam: anything that solves a LinearProblem will do."""
    calls = []

    def dense_solver(problem: LinearProblem) -> torch.Tensor:
        calls.append(problem.alpha)
        size = problem.rhs.numel()
        eye = torch.eye(size, dtype=problem.rhs.dtype)
        columns = torch.stack([problem.normal(row) for row in eye]).T
        weight = sum(term.weight for term in problem.scaled())
        return torch.linalg.solve(columns + weight * eye, problem.rhs)

    truth = torch.tensor([2.0, 1.5], dtype=torch.float64)
    found = gauss_newton(
        decay,
        decay(truth),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        solver=dense_solver,
        iterations=8,
        alpha=0.1,
    ).solution
    assert len(calls) == 8
    assert calls[1] == pytest.approx(calls[0] / 2)
    # alpha only reaches 0.1 / 2**7 in eight steps, so the answer is still
    # slightly held towards the reference. That is the schedule working.
    assert torch.allclose(found, truth, atol=1e-3)


def test_cg_solver_reaches_the_same_answer_as_the_direct_one() -> None:
    truth = torch.tensor([2.0, 1.5], dtype=torch.float64)
    start = torch.tensor([1.0, 1.0], dtype=torch.float64)
    data = decay(truth)
    iterative = gauss_newton(
        decay,
        data,
        start,
        solver=CGSolver(max_iter=100, rtol=1e-12),
        iterations=10,
        alpha=1e-2,
    ).solution
    direct = gauss_newton(
        decay, data, start, solver=LstsqSolver(), iterations=10, alpha=1e-2
    ).solution
    assert torch.allclose(iterative, direct, atol=1e-7)


def test_lstsq_solver_needs_the_jacobian() -> None:
    problem = LinearProblem(normal=lambda v: v, rhs=torch.zeros(3), alpha=0.1)
    with pytest.raises(ValueError, match="needs the Jacobian"):
        LstsqSolver()(problem)


def test_autodiff_derivative_and_adjoint_are_a_pair() -> None:
    generator = torch.Generator().manual_seed(4)
    model = autodiff(lambda x: torch.stack([x[0] ** 2, x[0] * x[1], x[1].sin()]))
    point = torch.tensor([0.7, -1.3], dtype=torch.float64)
    step = torch.randn(2, generator=generator, dtype=torch.float64)
    covector = torch.randn(3, generator=generator, dtype=torch.float64)
    left = torch.dot(model.derivative(point, step), covector)
    right = torch.dot(step, model.adjoint(point, covector))
    assert left == pytest.approx(right.item(), rel=1e-10)


def test_iterations_must_be_positive() -> None:
    with pytest.raises(ValueError, match="iterations"):
        gauss_newton(
            decay, decay(torch.ones(2).double()), torch.ones(2).double(), iterations=0
        )


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def batched_problem(voxels: int, rows: int = 24, columns: int = 3):
    """A batch of small independent least-squares problems, on the host."""
    generator = torch.Generator().manual_seed(5)
    jacobian = torch.randn(
        voxels, rows, columns, generator=generator, dtype=torch.float32
    )
    target = torch.randn(voxels, rows, generator=generator, dtype=torch.float32)
    penalty = torch.diff(torch.eye(columns, dtype=torch.float32), dim=0)
    return LinearProblem(
        normal=lambda v: v,
        rhs=torch.zeros(voxels, columns),
        alpha=0.2,
        regularizers=[
            Regularizer(1.0),
            Regularizer(
                0.5,
                operator=lambda v: penalty @ v,
                adjoint=lambda v: penalty.T @ v,
            ),
        ],
        matrix=jacobian,
        target=target,
    )


def test_streaming_is_not_used_for_a_small_batch() -> None:
    problem = batched_problem(16)
    assert LstsqSolver()._destination(problem.matrix) is None


@requires_cuda
def test_streamed_and_direct_agree() -> None:
    problem = batched_problem(9000)
    streamed = LstsqSolver(device="auto", chunk=1024)(problem)
    on_host = LstsqSolver(device=None)(problem)
    assert streamed.shape == on_host.shape
    assert streamed.device.type == "cpu"
    assert torch.allclose(streamed, on_host, atol=1e-4)


@requires_cuda
def test_a_ragged_final_chunk_is_handled() -> None:
    problem = batched_problem(5000)
    streamed = LstsqSolver(device="auto", chunk=1024)(problem)  # 4 full, one of 904
    on_host = LstsqSolver(device=None)(problem)
    assert torch.allclose(streamed, on_host, atol=1e-4)


@requires_cuda
def test_streaming_triggers_only_above_the_threshold() -> None:
    solver = LstsqSolver(stream_above=1000)
    assert solver._destination(batched_problem(2000).matrix) is not None
    assert solver._destination(batched_problem(500).matrix) is None
