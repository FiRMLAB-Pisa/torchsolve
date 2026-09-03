"""Memory-lean iterative solvers for inverse problems in PyTorch.

The solvers here take the normal operator rather than the forward one, because
that is what a non-Cartesian reconstruction can afford to apply repeatedly, and
they are written so that an iteration allocates nothing it does not have to.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._cg import CGResult, Regularizer, conjugate_gradient

try:
    __version__ = _distribution_version(__name__)
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "CGResult",
    "Regularizer",
    "__version__",
    "conjugate_gradient",
]
