from .model import NativeTypeNN, native_available

__all__ = ["NativeTypeNN", "native_available", "status"]


def status() -> str:
    ok, detail = native_available()
    return ("ready: " if ok else "unavailable: ") + detail
