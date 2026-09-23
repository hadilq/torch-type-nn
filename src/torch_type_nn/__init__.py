"""torch-type-nn: type-nn for PyTorch.

A type-nn layer maps x to z; output unit k is one And over its Ors r:

    Or_kr = w_kr . x + b_kr          sum type
    A_k   = prod_r Or_kr ^ a_kr      product type, a_kr >= 1
    z_k   = sign(A_k) ln(1 + |A_k|)  partition function

Reference implementation and design notes:
https://github.com/hadilq/type-nn, https://hadilq.com/posts/train-the-knowledge/
"""

from importlib.metadata import PackageNotFoundError, version

from . import functional
from .functional import and_or, readout
from .layer import AndOr
from .network import TypeNN, birth_depth
from .optim import TypeAdam
from .scaling import StructureScaler
from .train import FitResult, fit, mse_loss

try:
    __version__ = version("torch-type-nn")
except PackageNotFoundError:  # running from a checkout without install
    __version__ = "0.0.0+local"

__all__ = [
    "AndOr", "TypeNN", "TypeAdam", "StructureScaler", "fit", "FitResult", "mse_loss",
    "birth_depth", "and_or", "readout", "functional", "__version__",
]
