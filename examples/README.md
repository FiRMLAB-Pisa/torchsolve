# Examples

One example per capability, each checking the solver against something that
knows the answer. The `.py` is the source: it runs as a script, lints with the
rest of the package, and reads as a diff. The `.ipynb` beside it is generated
from it, executed, and committed with its outputs, so it opens in Colab and
runs top to bottom — there is nothing to download.

| example | shows | checked against |
|---|---|---|
| [`01-regularised_solve`](01-regularised_solve.ipynb) | `Regularizer`, its operator and its bias | the unregularised solve, and where the bias moves the answer |
| [`02-preconditioning`](02-preconditioning.ipynb) | `preconditioner=` | the same solve without one, iteration by iteration |
| [`03-differentiable_weight`](03-differentiable_weight.ipynb) | `parameters=`, and the implicit derivative | a sweep of 40 solves |
| [`04-irgnm`](04-irgnm.ipynb) | `gauss_newton`, its schedule, `solver=`, `batch_dims=` | the same fit at a fixed weight, and the truth |

[`figures/make_showcase.py`](figures/make_showcase.py) draws the README's
figure, and is not one of the examples.

## Rebuilding

```bash
pip install -e .[dev] jupytext nbclient ipykernel
bash scripts/build_examples.sh
```

Every notebook is regenerated from its script and executed against the
interpreter the package is installed into. `--check` verifies the notebooks are
current without running them.
