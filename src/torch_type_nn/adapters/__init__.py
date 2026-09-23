"""Adapters that put model families under the scaling protocol."""

from .mlp import MLPAdapter
from .typenn import TypeNNAdapter

__all__ = ["TypeNNAdapter", "MLPAdapter"]
