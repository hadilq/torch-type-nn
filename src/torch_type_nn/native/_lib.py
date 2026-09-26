from __future__ import annotations

import ctypes
import subprocess
from ctypes import POINTER, c_double, c_int, c_size_t, c_uint, c_void_p
from functools import lru_cache

from . import _build

TNN_PY_BIC, TNN_PY_THRESHOLD = 0, 1


class NativeUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load():
    try:
        path = _build.compile_library()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise NativeUnavailable(str(e)) from e
    lib = ctypes.CDLL(str(path))
    lib.tnn_py_create.argtypes = [c_int, c_size_t, c_size_t, c_uint]
    lib.tnn_py_create.restype = c_void_p
    lib.tnn_py_free.argtypes = [c_void_p, c_int]
    lib.tnn_py_begin.argtypes = [c_void_p, c_int, c_size_t, c_size_t, c_double]
    lib.tnn_py_end.argtypes = [c_void_p, c_int]
    lib.tnn_py_set_training.argtypes = [c_void_p, c_int, c_int]
    lib.tnn_py_forward.argtypes = [c_void_p, c_int, POINTER(c_double), POINTER(c_double)]
    lib.tnn_py_backward.argtypes = [c_void_p, c_int, POINTER(c_double)]
    lib.tnn_py_step.argtypes = [c_void_p, c_int, POINTER(c_double), POINTER(c_double)]
    lib.tnn_py_epoch.argtypes = [c_void_p, c_int, POINTER(c_double),
                                 POINTER(c_double), c_size_t, c_uint]
    lib.tnn_py_epoch_end.argtypes = [c_void_p, c_int]
    for name, rest in (("tnn_py_params", c_size_t), ("tnn_py_depth", c_size_t),
                       ("tnn_py_init_depth", c_size_t), ("tnn_py_n_in", c_size_t),
                       ("tnn_py_n_out", c_size_t), ("tnn_py_phase", c_int)):
        getattr(lib, name).argtypes = [c_void_p, c_int]
        getattr(lib, name).restype = rest
    lib.tnn_py_counters.argtypes = [c_void_p, c_int, POINTER(c_uint)]
    lib.tnn_py_structure.argtypes = [c_void_p, c_int, POINTER(c_int), c_int,
                                     POINTER(c_int), POINTER(c_int)]
    lib.tnn_py_structure.restype = c_int
    return lib


def native_available() -> tuple[bool, str]:
    try:
        return True, str(_build.compile_library())
    except Exception as e:
        return False, str(e)
