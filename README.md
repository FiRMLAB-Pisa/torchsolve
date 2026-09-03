# torchsolve

Iterative solvers for inverse problems in PyTorch, written to spend as little
memory per iteration as the arithmetic allows, and differentiable without
storing one.

[![Tests](https://github.com/FiRMLAB-Pisa/torchsolve/actions/workflows/test-ci.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/torchsolve/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/FiRMLAB-Pisa/torchsolve/branch/main/graph/badge.svg)](https://codecov.io/gh/FiRMLAB-Pisa/torchsolve)
[![PyPI](https://img.shields.io/pypi/v/torchsolve.svg)](https://pypi.org/project/torchsolve/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Conjugate gradient on the regularised normal equations,

$$\min_x \; \\|Ax - y\\|^2 + \\sum_k \\lambda_k \\|R_k x - c_k\\|^2 ,$$

solved as $\\big(A^HA + \\sum_k \\lambda_k R_k^H R_k\\big)x = A^Hy + \\sum_k
\\lambda_k R_k^H c_k$. The caller hands over $A^HA$ rather than $A$, because for
a non-Cartesian acquisition that operator is a convolution costing far less
than a transform pair, and because it is what a Toeplitz factorisation stands
for.

![what it does](examples/figures/showcase.png)

- **Any number of regularisation terms** — each with its own weight, its own
  linear operator (identity by default, and then folded into a single scalar
  rather than applied), and its own bias, which relocates what the term pulls
  towards
- **Preconditioning instead of density compensation** — the weighting belongs
  in the solver, where it changes only the path taken, rather than in the data,
  where it changes which problem is being solved
- **It steps through negative curvature** — a compressed Toeplitz normal
  carries eigenvalues just below zero, and a `pAp > 0` guard stops on them and
  leaves the residual where it was. BART's `conjgrad` stops only on an exactly
  zero curvature; so does this, and it says so in the result rather than
  failing quietly
- **Differentiable without unrolling** — a gradient reaching the solution
  reaches the right-hand side, and anything the operator closed over, through
  one more solve. Memory is flat in the iteration count, so a solve deep enough
  to converge costs a backward pass no larger than a shallow one
- **Four volumes, whatever the iteration count** — the updates are in place and
  the inner products are fused reductions

## Memory

Twenty iterations on a 192³ complex volume, one RTX 4060 Laptop GPU:

| | peak | |
|---|---|---|
| `torchsolve` | 216 MiB | **4.0 volumes**, 166 ms |
| the implementation it replaces | 432 MiB | 8.0 volumes, 254 ms |

The difference is entirely in how the arithmetic is written. `torch.vdot` and a
batched `einsum` take an inner product without materialising it;
`(a.conj() * b).real.sum()` costs two whole volumes and `torch.linalg.vecdot`
costs the same. The updates are `addcmul_` and `mul_`, not `x = x + a * p`.
What is left is the four the algorithm needs — solution, residual, direction,
and whatever the operator itself returns.

## Quick Start

```bash
pip install torchsolve
```

```python
import torch
from torchsolve import Regularizer, conjugate_gradient

# the plain solve: hand over the normal operator and A^H y
result = conjugate_gradient(normal, rhs, max_iter=20, rtol=1e-6)
result.solution, result.iterations, result.converged, result.definite

# weight the iteration rather than the data
result = conjugate_gradient(normal, rhs, preconditioner=lambda r: r / diagonal)

# any number of terms: weight, operator (identity by default), bias
result = conjugate_gradient(
    normal,
    rhs,
    regularizers=[
        Regularizer(1e-3),                                  # towards zero
        Regularizer(2e-2, bias=previous_estimate),          # towards a prior
        Regularizer(5e-2, operator=gradient, adjoint=divergence),  # smoothness
    ],
)

# a batch of independent systems, each taking its own step size
result = conjugate_gradient(normal, rhs, batch_dim=0)

# learn something inside the operator, differentiating through the solve
result = conjugate_gradient(normal, rhs, parameters=[weight])
result.solution.pow(2).sum().backward()
```

## Examples

| | |
|---|---|
| [`preconditioning.py`](examples/preconditioning.py) | A badly scaled operator, and what Jacobi buys without touching the data |
| [`regularised_solve.py`](examples/regularised_solve.py) | A shaped penalty and a bias, on a noisy staircase |
| [`differentiable_weight.py`](examples/differentiable_weight.py) | Learning the regularisation weight, checked against a sweep |
| [`make_showcase.py`](examples/make_showcase.py) | The figure above |

## Related Works

- **BART** — <https://mrirecon.github.io/bart/>. Its `conjgrad` in
  `src/iter/italgos.c` is the reference this follows, including the decision to
  stop only on an exactly zero curvature. `tests/test_cg.py` transcribes it and
  checks the two agree.
- **MIRTorch** — <https://github.com/guanhuaw/MIRTorch>. Where the
  differentiate-the-solve-not-the-iterations approach comes from. Its CG gates
  on positive curvature, which is the behaviour this deliberately does not
  copy.
- **SigPy** — <https://github.com/mikgroup/sigpy>. `sigpy.alg.ConjugateGradient`
  for the same problem in NumPy and CuPy.
- Hestenes MR, Stiefel E. *Methods of conjugate gradients for solving linear
  systems.* J Res Natl Bur Stand 1952;49:409-436.
- Pruessmann KP, Weiger M, Börnert P, Boesiger P. *Advances in sensitivity
  encoding with arbitrary k-space trajectories.* Magn Reson Med 2001;46:638-651.
  Why a non-Cartesian reconstruction wants the normal operator and a
  preconditioner rather than a density-weighted adjoint.

## Development

```bash
pip install -e .[dev]
bash scripts/format_and_lint.sh
pytest -q
```

The docstring examples run as part of the suite — they are the documentation,
and an example that has drifted is a broken one. See
[CONTRIBUTING.md](CONTRIBUTING.md).
