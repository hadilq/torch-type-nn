from __future__ import annotations

BACKENDS = ("torch", "c", "cuda")


def available_backends() -> dict[str, str]:
    from . import cuda, native
    return {
        "torch": "padded (m, R, n) tensors; batched; in-tree",
        "c": native.status(),
        "cuda": cuda.status(),
    }


def get_backend(name: str = "torch"):
    key = str(name).lower()
    if key == "torch":
        from .network import TypeNN
        return TypeNN
    if key in ("c", "native", "type-nn-c"):
        from .native import NativeTypeNN
        return NativeTypeNN
    if key in ("cuda", "gpu"):
        from .cuda import CudaTypeNN
        return CudaTypeNN
    raise ValueError(f"unknown backend {name!r}; choose one of {BACKENDS}")
