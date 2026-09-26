"""Compile ``libtnn_py.so`` from the vendored C sources (or ``TYPE_NN_SRC``)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDORED = _HERE / "c"
_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torch-type-nn"


def source_dir() -> Path:
    env = os.environ.get("TYPE_NN_SRC")
    if env:
        p = Path(env)
        if (p / "type_nn.c").exists():
            return p
    return _VENDORED


def library_path() -> Path:
    ext = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    override = os.environ.get("TNN_PY_LIB")
    if override:
        return Path(override)
    return _CACHE / f"libtnn_py{ext}"


def compile_library(force: bool = False) -> Path:
    """Return the path to a loadable ``libtnn_py``. Compiles if needed."""
    dest = library_path()
    src = source_dir()
    cc = shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")
    if not cc:
        raise FileNotFoundError("no C compiler (cc/gcc) on PATH; cannot build the native backend")
    needed = [
        _VENDORED / "tnn_py.c",
        src / "type_nn.c",
        src / "type_nn_scale.c",
        src / "type_nn_overfit.c",
        src / "type_nn_overfit_scale.c",
    ]
    for f in needed:
        if not f.exists():
            raise FileNotFoundError(f"native backend source missing: {f}")
    if dest.exists() and not force:
        newest_src = max(f.stat().st_mtime for f in needed)
        if dest.stat().st_mtime >= newest_src:
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    includes = ["-I" + str(_VENDORED), "-I" + str(src)]
    cmd = [
        cc, "-O2", "-std=c11", "-shared", "-fPIC",
        *includes,
        str(_VENDORED / "tnn_py.c"),
        str(src / "type_nn.c"),
        str(src / "type_nn_scale.c"),
        str(src / "type_nn_overfit.c"),
        str(src / "type_nn_overfit_scale.c"),
        "-lm", "-o", str(dest),
    ]
    subprocess.run(cmd, check=True)
    return dest
