"""C graph-type-nn (``type_nn_graph.c``) behind the stable ``tnn_py_g_*`` ABI.

``NativeGraphTypeNN`` is the reference: the same object the C board runs,
driven by the same bench.c protocol (per-sample mean-MSE step, xorshift
shuffle per epoch). ``snapshot()`` reads its whole state (nodes, units, Ors
with their Adam moments, edges with their weights and feedback, counters
and scalars) so the Python and device ports can be checked against it
number for number, or started from it (``GraphTypeNN.from_snapshot``).
"""

from __future__ import annotations

import ctypes
from ctypes import c_char, c_double, c_long, c_size_t, c_uint

import torch

from . import _lib

__all__ = ["NativeGraphTypeNN"]

_DP = ctypes.POINTER(c_double)


def _cpu64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float64, device="cpu").contiguous()
    return torch.as_tensor(x, dtype=torch.float64).contiguous()


def _ptr(t: torch.Tensor):
    return ctypes.cast(t.data_ptr(), _DP)


class NativeGraphTypeNN:
    """C graph-type-nn. Board row **C graph-type-nn**."""

    name = "c-graph"

    def __init__(self, in_features: int, out_features: int, *, seed: int = 1) -> None:
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.seed = int(seed)
        self._lib = _lib.load()
        self._ptr = self._lib.tnn_py_g_create(self.in_features, self.out_features,
                                              c_uint(self.seed & 0xFFFFFFFF))
        if not self._ptr:
            raise _lib.NativeUnavailable("tnn_py_g_create returned NULL")
        self._begun = False

    def __del__(self) -> None:
        ptr, lib = getattr(self, "_ptr", None), getattr(self, "_lib", None)
        if ptr and lib is not None:
            lib.tnn_py_g_free(ptr)
            self._ptr = None

    # ------------------------------------------------------------ accounting

    def _sizes(self) -> list[int]:
        s = (c_size_t * 6)()
        self._lib.tnn_py_g_sizes(self._ptr, s)
        return [int(v) for v in s]

    @property
    def n_nodes(self) -> int:
        return self._sizes()[2]

    @property
    def n_edges(self) -> int:
        return self._sizes()[3]

    @property
    def depth(self) -> int:
        """Nodes, the board's ``layers`` column."""
        return self.n_nodes

    @property
    def birth_depth(self) -> int:
        return self._sizes()[4]

    @property
    def birth_edges(self) -> int:
        return self._sizes()[5]

    def num_params(self) -> int:
        return int(self._lib.tnn_py_g_params(self._ptr))

    def counters(self) -> dict[str, int]:
        raw = (c_uint * 8)()
        self._lib.tnn_py_g_counters(self._ptr, raw)
        names = ("or_add", "or_drop", "and_add", "and_drop",
                 "node_add", "node_drop", "edge_add", "edge_drop")
        out = {k: int(raw[i]) for i, k in enumerate(names)}
        out["layer_add"], out["layer_drop"] = out["node_add"], out["node_drop"]
        return out

    def structure(self) -> dict:
        snap = self.snapshot(weights=False)
        return {"nodes": [[len(u["ors"]) for u in n["units"]] for n in snap["nodes"]],
                "edges": [(e["src"], e["dst"], len(e["pairs"])) for e in snap["edges"]]}

    def check(self) -> tuple[bool, str]:
        buf = (c_char * 256)()
        ok = bool(self._lib.tnn_py_g_check(self._ptr, buf, 256))
        return ok, buf.value.decode()

    # --------------------------------------------------------------- protocol

    def begin(self, n_train: int, epochs: int, lr: float) -> None:
        self._lib.tnn_py_g_begin(self._ptr, int(n_train), int(epochs), float(lr))
        self._begun = True

    def end(self) -> None:
        if self._begun:
            self._lib.tnn_py_g_end(self._ptr)

    def epoch_end(self) -> None:
        self._lib.tnn_py_g_epoch_end(self._ptr)

    def train(self, mode: bool = True) -> "NativeGraphTypeNN":
        self._lib.tnn_py_g_set_training(self._ptr, int(bool(mode)))
        return self

    def eval(self) -> "NativeGraphTypeNN":
        return self.train(False)

    def forward(self, x) -> torch.Tensor:
        t = _cpu64(x)
        squeeze = t.ndim == 1
        if squeeze:
            t = t.unsqueeze(0)
        if t.shape[-1] != self.in_features:
            raise ValueError(f"expected n_in={self.in_features}, got {tuple(t.shape)}")
        y = torch.empty(t.shape[0], self.out_features, dtype=torch.float64)
        for i in range(t.shape[0]):
            self._lib.tnn_py_g_forward(self._ptr, _ptr(t[i]), _ptr(y[i]))
        return y[0] if squeeze else y

    __call__ = forward

    def backward(self, dy) -> None:
        d = _cpu64(dy).reshape(-1)
        self._lib.tnn_py_g_backward(self._ptr, _ptr(d))

    def step(self, x, t) -> None:
        """One bench.c sample: forward + mean-MSE backward + Adam, in C."""
        x64, t64 = _cpu64(x).reshape(-1), _cpu64(t).reshape(-1)
        self._lib.tnn_py_g_step(self._ptr, _ptr(x64), _ptr(t64))

    def epoch(self, X, Y, shuffle_seed: int) -> None:
        """One bench.c epoch: xorshift shuffle, training steps, epoch_end."""
        X64, Y64 = _cpu64(X), _cpu64(Y)
        self._lib.tnn_py_g_epoch(self._ptr, _ptr(X64), _ptr(Y64), X64.shape[0],
                                 c_uint(shuffle_seed & 0xFFFFFFFF))

    # ------------------------------------------------------------ the state

    def scalars(self) -> dict:
        s = (c_double * 12)()
        self._lib.tnn_py_g_scalars(self._ptr, s)
        keys = ("lr", "step", "total", "phase", "crit_best", "epoch_loss", "epoch_base",
                "rng", "cache_n", "training", "n_train", "epochs")
        out = {k: float(s[i]) for i, k in enumerate(keys)}
        for k in ("step", "total", "phase", "rng", "cache_n", "training", "n_train", "epochs"):
            out[k] = int(out[k])
        return out

    def snapshot(self, weights: bool = True) -> dict:
        """Everything the C graph holds, as plain Python / torch values."""
        lib, g = self._lib, self._ptr
        sizes = self._sizes()
        nodes = []
        for v in range(sizes[2]):
            info, fb = (c_long * 4)(), (c_double * 6)()
            lib.tnn_py_g_node(g, v, info, fb)
            n_in, n_out = int(info[0]), int(info[1])
            units = []
            for k in range(n_out):
                ui = (c_long * 3)()
                lib.tnn_py_g_unit(g, v, k, ui)
                ors = []
                for r in range(int(ui[0])):
                    w = torch.zeros(n_in, dtype=torch.float64)
                    mw, vw = torch.zeros_like(w), torch.zeros_like(w)
                    sc = (c_double * 11)()
                    lib.tnn_py_g_or(g, v, k, r, _ptr(w) if weights else None,
                                    _ptr(mw) if weights else None,
                                    _ptr(vw) if weights else None, sc)
                    ors.append({"w": w, "mw": mw, "vw": vw, "b": sc[0], "mb": sc[1],
                                "vb": sc[2], "a": sc[3], "ma": sc[4], "va": sc[5],
                                "t": int(sc[6]), "born": int(sc[7]), "probe": bool(sc[8]),
                                "gb": sc[9], "ga": sc[10]})
                units.append({"ors": ors, "probe": bool(ui[1]), "born": int(ui[2])})
            nodes.append({"n_in": n_in, "n_out": n_out, "probe": bool(info[2]),
                          "born": int(info[3]), "units": units,
                          "feedback": [float(x) for x in fb]})
        edges = []
        for e in range(sizes[3]):
            info, d = (c_long * 5)(), (c_double * 11)()
            lib.tnn_py_g_edge(g, e, info, d)
            npairs = int(info[2])
            ks, js = (c_long * max(npairs, 1))(), (c_long * max(npairs, 1))()
            lib.tnn_py_g_edge_pairs(g, e, ks, js)
            edges.append({"src": int(info[0]), "dst": int(info[1]), "probe": bool(info[3]),
                          "born": int(info[4]),
                          "pairs": [(int(ks[q]), int(js[q])) for q in range(npairs)],
                          "in_w": d[0], "out_w": d[1], "c": d[2],
                          "feedback": [float(x) for x in d[3:]]})
        return {"n_in": sizes[0], "n_out": sizes[1], "birth_vertices": sizes[4],
                "birth_edges": sizes[5], "nodes": nodes, "edges": edges,
                "counters": self.counters(), "scalars": self.scalars()}
