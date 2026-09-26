"""C type-nn / type-nn-overfit behind a Python interface.

The C models keep one weight vector per live Or (length ``n_in``). There
is no padded ``R`` and no ``mask``. That is the memory-scaling path the
architecture asked for; the torch ``AndOr`` keeps the padded layout for
batched GEMMs.

Build the shared library on first use (needs a C compiler). Override the
sources with ``TYPE_NN_SRC`` if you want a different checkout than the
vendored snapshot.
"""

from __future__ import annotations

from .model import NativeTypeNN, native_available

__all__ = ["NativeTypeNN", "native_available", "status"]


def status() -> str:
    ok, detail = native_available()
    return ("ready: " if ok else "unavailable: ") + detail
