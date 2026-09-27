from .model import CudaTypeNN, cuda_available

__all__ = ["CudaTypeNN", "cuda_available", "status"]


def status() -> str:
    ok, detail = cuda_available()
    return ("ready: " if ok else "unavailable: ") + detail
