"""ctypes binding to ``libtnn_py``."""

from __future__ import annotations

import ctypes
import subprocess
from ctypes import POINTER, c_double, c_int, c_uint, c_void_p, c_size_t
from functools import lru_cache

from . import _build

TNN_PY_BIC = 0
TNN_PY_THRESHOLD = 1


class NativeUnavailable(RuntimeError):
    """The C library could not be built or loaded."""


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
    lib.tnn_py_epoch_end.argtypes = [c_void_p, c_int]
    lib.tnn_py_params.argtypes = [c_void_p, c_int]
    lib.tnn_py_params.restype = c_size_t
    lib.tnn_py_depth.argtypes = [c_void_p, c_int]
    lib.tnn_py_depth.restype = c_size_t
    lib.tnn_py_init_depth.argtypes = [c_void_p, c_int]
    lib.tnn_py_init_depth.restype = c_size_t
    lib.tnn_py_n_in.argtypes = [c_void_p, c_int]
    lib.tnn_py_n_in.restype = c_size_t
    lib.tnn_py_n_out.argtypes = [c_void_p, c_int]
    lib.tnn_py_n_out.restype = c_size_t
    lib.tnn_py_phase.argtypes = [c_void_p, c_int]
    lib.tnn_py_phase.restype = c_int
    lib.tnn_py_counters.argtypes = [c_void_p, c_int, POINTER(c_uint)]
    lib.tnn_py_structure.argtypes = [c_void_p, c_int, POINTER(c_int), c_int,
                                     POINTER(c_int), POINTER(c_int)]
    lib.tnn_py_structure.restype = c_int
    return lib


def native_available() -> tuple[bool, str]:
    try:
        path = _build.compile_library()
    except Exception as e:
        return False, str(e)
    return True, str(path)
