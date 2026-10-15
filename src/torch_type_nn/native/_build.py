"""Compile libtnn_py from native/c (checkout or wheel) or TYPE_NN_SRC."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def bridge_dir() -> Path:
    """Directory holding tnn_py.c: package native/c, else a leftover repo-root csrc/."""
    for cand in (_HERE / "c", _HERE.parents[2] / "csrc"):
        if (cand / "tnn_py.c").exists():
            return cand
    raise FileNotFoundError("tnn_py.c not found (expected at torch_type_nn/native/c)")


def type_nn_dir() -> Path:
    """One tree only. Mixing TYPE_NN_SRC .c with vendored headers drifts."""
    env = os.environ.get("TYPE_NN_SRC")
    if (env and Path(env, "type_nn.c").exists() and Path(env, "tnn_py.c").exists()
            and Path(env, "type_nn_graph.c").exists()):
        return Path(env)
    return bridge_dir()


def library_path() -> Path:
    ext = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    if os.environ.get("TNN_PY_LIB"):
        return Path(os.environ["TNN_PY_LIB"])
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torch-type-nn"
    return cache / f"libtnn_py{ext}"


def compile_library(force: bool = False) -> Path:
    dest = library_path()
    src, bridge = type_nn_dir(), bridge_dir()
    cc = shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")
    if not cc:
        raise FileNotFoundError("no C compiler (cc/gcc) on PATH")
    files = [
        bridge / "tnn_py.c",
        src / "type_nn.c", src / "type_nn_scale.c",
        src / "type_nn_overfit.c", src / "type_nn_overfit_scale.c",
        src / "type_nn_graph.c",
    ]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"native source missing: {f}")
    newest = max(f.stat().st_mtime for f in files)

    def fresh() -> bool:
        return not force and dest.exists() and dest.stat().st_mtime >= newest

    if fresh():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Parallel processes (bench --jobs, pytest-xdist) can all find the library
    # stale at once. A lock serialises the compile, and an atomic rename means
    # a reader never dlopens a half-written file.
    with _lock(dest.with_name(dest.name + ".lock")):
        if fresh():
            return dest
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        try:
            subprocess.run(
                [cc, "-O2", "-std=c11", "-shared", "-fPIC",
                 f"-I{bridge}", f"-I{src}", *[str(f) for f in files],
                 "-lm", "-o", str(tmp)],
                check=True,
            )
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
    return dest


@contextlib.contextmanager
def _lock(path: Path):
    try:
        import fcntl
    except ImportError:  # Windows: the atomic rename alone keeps readers safe
        yield
        return
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
