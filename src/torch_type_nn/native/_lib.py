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
    _graph_signatures(lib)
    return lib


def _graph_signatures(lib) -> None:
    from ctypes import POINTER, c_char_p, c_long
    P = POINTER
    lib.tnn_py_g_create.argtypes = [c_size_t, c_size_t, c_uint]
    lib.tnn_py_g_create.restype = c_void_p
    lib.tnn_py_g_free.argtypes = [c_void_p]
    lib.tnn_py_g_begin.argtypes = [c_void_p, c_size_t, c_size_t, c_double]
    lib.tnn_py_g_end.argtypes = [c_void_p]
    lib.tnn_py_g_set_training.argtypes = [c_void_p, c_int]
    lib.tnn_py_g_forward.argtypes = [c_void_p, P(c_double), P(c_double)]
    lib.tnn_py_g_backward.argtypes = [c_void_p, P(c_double)]
    lib.tnn_py_g_step.argtypes = [c_void_p, P(c_double), P(c_double)]
    lib.tnn_py_g_epoch.argtypes = [c_void_p, P(c_double), P(c_double), c_size_t, c_uint]
    lib.tnn_py_g_epoch_end.argtypes = [c_void_p]
    lib.tnn_py_g_params.argtypes = [c_void_p]
    lib.tnn_py_g_params.restype = c_size_t
    lib.tnn_py_g_sizes.argtypes = [c_void_p, P(c_size_t)]
    lib.tnn_py_g_counters.argtypes = [c_void_p, P(c_uint)]
    lib.tnn_py_g_scalars.argtypes = [c_void_p, P(c_double)]
    lib.tnn_py_g_node.argtypes = [c_void_p, c_size_t, P(c_long), P(c_double)]
    lib.tnn_py_g_node.restype = c_int
    lib.tnn_py_g_unit.argtypes = [c_void_p, c_size_t, c_size_t, P(c_long)]
    lib.tnn_py_g_unit.restype = c_int
    lib.tnn_py_g_or.argtypes = [c_void_p, c_size_t, c_size_t, c_size_t,
                                P(c_double), P(c_double), P(c_double), P(c_double)]
    lib.tnn_py_g_or.restype = c_int
    lib.tnn_py_g_edge.argtypes = [c_void_p, c_size_t, P(c_long), P(c_double)]
    lib.tnn_py_g_edge.restype = c_int
    lib.tnn_py_g_edge_pairs.argtypes = [c_void_p, c_size_t, P(c_long), P(c_long)]
    lib.tnn_py_g_edge_pairs.restype = c_int
    lib.tnn_py_g_check.argtypes = [c_void_p, c_char_p, c_size_t]
    lib.tnn_py_g_check.restype = c_int


def native_available() -> tuple[bool, str]:
    try:
        return True, str(_build.compile_library())
    except Exception as e:
        return False, str(e)
