"""Where a type-nn lives: torch tensors, the C reference, or (later) CUDA.

The PyTorch ``AndOr`` layer is a *padded* ``(m, R, n)`` tensor so a batch is
one GEMM. That is the right trade for GPU throughput; it is the wrong
trade for the architecture's other goal — a memory footprint that grows
and shrinks with the live Ors.

A *backend* is the same network with a different store:

* ``"torch"`` — ``TypeNN`` / ``AndOr`` (padded, batched, differentiable).
* ``"c"``     — the C reference: one ``double *w`` of length ``n_in`` per
  live Or, no mask. Per-sample. This is the small-footprint path.
* ``"cuda"``  — reserved. Same ragged layout on device; not shipped yet.

Pick one at construction. They do not share weights.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Backend", "available_backends", "get_backend"]

BACKENDS = ("torch", "c", "cuda")


@runtime_checkable
class Backend(Protocol):
    """Minimum a type-nn store has to expose."""

    name: str

    def describe(self) -> str:
        """One line: layout, batching, what is missing."""


def available_backends() -> dict[str, str]:
    """Name → status. ``cuda`` is listed so a later implementation can
    register itself without changing the public names."""
    from . import native
    out = {
        "torch": "padded (m, R, n) tensors; batched; in-tree",
        "c": native.status(),
        "cuda": "not implemented (ragged device store; see docs/BACKENDS.md)",
    }
    return out


def get_backend(name: str = "torch"):
    """Return a model factory for ``name``.

    * ``"torch"`` → :class:`~torch_type_nn.network.TypeNN`
    * ``"c"``     → :class:`~torch_type_nn.native.NativeTypeNN`
    * ``"cuda"``  → raises ``NotImplementedError`` until a device store lands
    """
    key = str(name).lower()
    if key == "torch":
        from .network import TypeNN
        return TypeNN
    if key in ("c", "native", "type-nn-c"):
        from .native import NativeTypeNN
        return NativeTypeNN
    if key == "cuda":
        raise NotImplementedError(
            "the CUDA backend is reserved: a ragged device store with the "
            "same live-Or layout as the C reference, not a port of the "
            "padded torch tensors. See docs/BACKENDS.md."
        )
    raise ValueError(f"unknown backend {name!r}; choose one of {BACKENDS}")
