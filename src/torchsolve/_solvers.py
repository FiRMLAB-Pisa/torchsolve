"""Inner solvers: one iterative, one direct.

Which to use follows from the size of the step, not from taste. An iterative
solver needs only to apply the normal operator, so it is what a matrix-free
problem has; a direct one factorises the Jacobian, which is faster and exact
when the Jacobian is small enough to write down -- a per-voxel model fit, where
there are millions of tiny independent problems rather than one large one.
"""

from __future__ import annotations

from collections.abc import Sequence
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

        \min_z \Big\| \begin{bmatrix} J \\ \sqrt{\alpha w_1} R_1 \\
        \vdots \end{bmatrix} z - \begin{bmatrix} r \\ \sqrt{\alpha w_1} c_1
        \\ \vdots \end{bmatrix} \Big\|^2

    Stacking rather than forming :math:`J^H J + \alpha R^H R` squares nothing,
    so the factorisation sees the conditioning of the problem rather than its
    square. This is what generalises :func:`torch.linalg.lstsq`: the same solve,
    with any number of regularisation operators, their own relative weights and
    their own biases.

    Leading axes of the Jacobian are a batch, so a per-voxel fit solves every
    voxel in one call. A batch that lives on the host and is large enough to be
    worth it is sent to the GPU in chunks, with the next chunk's upload
    overlapping the current one's solve, and only the raw Jacobian crosses --
    the stacked system is assembled on the device, so the enlarged matrix is
    never held on the host.

    Parameters
    ----------
    driver
        Passed to :func:`torch.linalg.lstsq`. ``None`` lets torch choose:
        ``gelsy`` on the host, ``gels`` on a device.
    device
        Where to solve. ``"auto"`` sends a host-resident batch to CUDA when one
        is available and the batch is worth the transfer; ``None`` solves
        wherever the problem already is.
    chunk
        Problems per chunk. ``None`` sends the batch in one piece, which is
        what to do whenever it fits; set it when it does not.
    stream_above
        Batch size below which streaming costs more than it saves.

    Examples
    --------
    >>> from torchsolve import LstsqSolver
    >>> LstsqSolver().driver is None
    True
    """

    driver: str | None = None
    device: str | torch.device | None = "auto"
    chunk: int | None = None
    stream_above: int = 4096
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
        terms = problem.scaled()
        destination = self._destination(jacobian)
        if destination is None:
            return self._stacked_solve(jacobian, target, terms)
        return self._streamed_solve(jacobian, target, terms, destination)

    def _destination(self, jacobian: torch.Tensor) -> torch.device | None:
        """Where to send the work, or ``None`` to leave it where it is."""
        if self.device is None:
            return None
        if self.device != "auto":
            wanted = torch.device(self.device)
            return None if wanted == jacobian.device else wanted
        if jacobian.device.type != "cpu" or not torch.cuda.is_available():
            return None
        if jacobian.ndim < 3 or _batch_size(jacobian) < self.stream_above:
            return None
        return torch.device("cuda")

    def _blocks(
        self,
        terms: Sequence[Regularizer],
        columns: int,
        like: torch.Tensor,
        destination: torch.device | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """Build the regularisation rows and their targets, unbatched.

        They do not depend on the batch, so a streamed solve builds them once
        and broadcasts them over every chunk. They are built where the problem
        already is, because a regularizer's operator is the caller's and may
        close over tensors that live on the host, and only then moved.
        """
        rows: list[torch.Tensor] = []
        biases: list[torch.Tensor | None] = []
        for term in terms:
            scale = term.weight**0.5
            if term.is_identity:
                eye = torch.eye(columns, dtype=like.dtype, device=like.device)
                rows.append(eye * scale)
            else:
                rows.append(_probe(term, columns, like) * scale)
            bias = term.bias
            biases.append(None if bias is None else bias.to(like.device) * scale)
        if destination is not None:
            rows = [row.to(destination) for row in rows]
            biases = [None if b is None else b.to(destination) for b in biases]
        return rows, biases

    def _stacked_solve(
        self,
        jacobian: torch.Tensor,
        target: torch.Tensor,
        terms: Sequence[Regularizer],
    ) -> torch.Tensor:
        """Stack and factorise, on whatever device the pieces are already on."""
        columns = jacobian.shape[-1]
        rows, biases = self._blocks(terms, columns, jacobian)
        leading = jacobian.shape[:-2]
        blocks = [jacobian]
        pieces = [target]
        for block, bias in zip(rows, biases, strict=True):
            blocks.append(block.expand(*leading, *block.shape))
            if bias is None:
                pieces.append(
                    torch.zeros(
                        *leading,
                        block.shape[-2],
                        dtype=target.dtype,
                        device=target.device,
                    )
                )
            else:
                pieces.append(bias.expand(*leading, bias.shape[-1]))
        stacked = torch.cat(blocks, dim=-2)
        answer = torch.cat(pieces, dim=-1)
        return torch.linalg.lstsq(
            stacked, answer.unsqueeze(-1), driver=self.driver
        ).solution.squeeze(-1)

    def _streamed_solve(
        self,
        jacobian: torch.Tensor,
        target: torch.Tensor,
        terms: Sequence[Regularizer],
        destination: torch.device,
    ) -> torch.Tensor:
        """Solve on the device, in chunks when asked, and bring the answer back.

        Chunking is for fitting, not for speed. Overlapping each chunk's upload
        with the previous chunk's solve was written and measured, and removed:
        it lost to plain sequential chunking at every size tried, by 0.13x to
        0.79x from pageable memory and still by 0.27x to 0.91x from pinned
        memory where no staging copy is needed at all. The factorisation
        appears to synchronise internally, which would serialise the streams
        while still charging for them.
        """
        flat_jacobian = jacobian.reshape(-1, *jacobian.shape[-2:])
        flat_target = target.reshape(-1, target.shape[-1])
        total, _rows, columns = flat_jacobian.shape
        size = self.chunk or total
        blocks, biases = self._blocks(terms, columns, jacobian, destination)

        answer = torch.empty(total, columns, dtype=jacobian.dtype)
        for start in range(0, total, size):
            stop = min(start + size, total)
            solved = self._chunk_solve(
                flat_jacobian[start:stop].to(destination),
                flat_target[start:stop].to(destination),
                blocks,
                biases,
            )
            answer[start:stop] = solved.to("cpu")
        return answer.reshape(*jacobian.shape[:-2], columns)

    def _chunk_solve(
        self,
        jacobian: torch.Tensor,
        target: torch.Tensor,
        blocks: Sequence[torch.Tensor],
        biases: Sequence[torch.Tensor | None],
    ) -> torch.Tensor:
        """Assemble the stacked system on the device and factorise it."""
        width = jacobian.shape[0]
        stacked = [jacobian]
        answer = [target]
        for block, bias in zip(blocks, biases, strict=True):
            stacked.append(block.expand(width, *block.shape))
            if bias is None:
                answer.append(
                    torch.zeros(
                        width,
                        block.shape[-2],
                        dtype=target.dtype,
                        device=target.device,
                    )
                )
            else:
                answer.append(bias.expand(width, bias.shape[-1]))
        return torch.linalg.lstsq(
            torch.cat(stacked, dim=-2),
            torch.cat(answer, dim=-1).unsqueeze(-1),
            driver=self.driver,
        ).solution.squeeze(-1)


def _probe(term: Regularizer, columns: int, like: torch.Tensor) -> torch.Tensor:
    """Apply a regularizer's operator to a basis, wherever that operator works.

    The operator is the caller's and may close over tensors on either the host
    or a device, and which is not knowable from here. Try where the problem is,
    and fall back to the host, since an operator built alongside host data is
    the case that surprises people.
    """
    for device in (like.device, torch.device("cpu")):
        eye = torch.eye(columns, dtype=like.dtype, device=device)
        try:
            applied = torch.stack([term.operator(row) for row in eye])  # type: ignore[misc]
        except RuntimeError as error:  # pragma: no cover - device-specific
            if "same device" not in str(error) or device.type == "cpu":
                raise
            continue
        return applied.transpose(-2, -1).to(like.device)
    raise RuntimeError("unreachable")  # pragma: no cover


def _batch_size(jacobian: torch.Tensor) -> int:
    """Return how many independent problems the leading axes hold."""
    total = 1
    for extent in jacobian.shape[:-2]:
        total *= extent
    return total
