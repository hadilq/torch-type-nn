"""graph-type-nn on a torch device (CUDA when present): ``CudaGraphTypeNN``.

Each node packs its live Ors into ``W (R, n_in)`` (C's memory layout, no
padding): the forward is one GEMM ``X @ W.T + b`` and a segmented product
per unit; the per-sample backward and the per-Or Adam step are written by
hand, as in ``type_nn.c`` (including the cofactor at an exact zero factor and
the ``lr / c`` step of a column wrapped by edge weight ``c``). The rule is the
shared port of ``type_nn_graph.c`` (:mod:`torch_type_nn.graph.core`), so the
structure follows the C graph's; the evidence cache lives on the device and
every ablation is one batched forward.

Without a GPU the same store runs on the CPU, which is how CI covers it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..graph.core import LR_SCALE, MASK, GraphCore, NodeStore, xorshift32

__all__ = ["CudaGraphTypeNN", "PackedStore"]

B1, B2, EPS = 0.9, 0.999, 1e-8


class PackedStore(NodeStore):
    """One node: live Ors packed by unit (C order), with their Adam state."""

    def __init__(self, n_in: int, n_out: int, device, dtype) -> None:
        self._n_in = int(n_in)
        self.device, self.dtype = torch.device(device), dtype
        f = {"device": self.device, "dtype": dtype}
        self.W = torch.zeros(0, n_in, **f)
        self.mW = torch.zeros(0, n_in, **f)
        self.vW = torch.zeros(0, n_in, **f)
        for name in ("b", "mb", "vb", "a", "ma", "va"):
            setattr(self, name, torch.zeros(0, **f))
        self.t = torch.zeros(0, dtype=torch.long, device=self.device)
        self.born = torch.zeros(0, dtype=torch.long, device=self.device)
        self.probe = torch.zeros(0, dtype=torch.bool, device=self.device)
        self.ptr = [0] * (n_out + 1)
        self.uprobe = [False] * n_out
        self.uborn = [0] * n_out
        self._uid = None
        self._x = self._o = None

    # ------------------------------------------------------------- sizes

    @property
    def n_in(self) -> int:
        return self._n_in

    @property
    def n_out(self) -> int:
        return len(self.ptr) - 1

    def n_or(self, k):
        return self.ptr[k + 1] - self.ptr[k]

    def total_ors(self):
        return int(self.W.shape[0])

    def num_params(self):
        return self.total_ors() * (self.n_in + 2)

    def uid(self) -> torch.Tensor:
        if self._uid is None:
            counts = torch.tensor([self.n_or(k) for k in range(self.n_out)],
                                  dtype=torch.long, device=self.device)
            self._uid = torch.repeat_interleave(
                torch.arange(self.n_out, device=self.device), counts)
        return self._uid

    def _rows_changed(self):
        self._uid = None
        self._x = self._o = None

    # --------------------------------------------------------------- Ors

    def get_or(self, k, r):
        i = self.ptr[k] + r
        return (self.W[i].detach().to(torch.float64).cpu().clone(),
                float(self.b[i]), float(self.a[i]))

    def set_or(self, k, r, w=None, b=None, a=None):
        i = self.ptr[k] + r
        if w is not None:
            self.W[i] = w.to(self.W)
        if b is not None:
            self.b[i] = b
        if a is not None:
            self.a[i] = a

    def or_probe(self, k, r):
        return bool(self.probe[self.ptr[k] + r])

    def set_or_probe(self, k, r, v):
        self.probe[self.ptr[k] + r] = bool(v)

    def or_born(self, k, r):
        return int(self.born[self.ptr[k] + r])

    def _insert_rows(self, at: int, w: torch.Tensor, b: float, a: float,
                     probe: bool, born: int) -> None:
        f = {"device": self.device, "dtype": self.dtype}
        row = w.to(**f).reshape(1, -1)
        z = torch.zeros(1, self.n_in, **f)
        s = torch.zeros(1, **f)

        def ins(t, v):
            return torch.cat([t[:at], v, t[at:]])
        self.W, self.mW, self.vW = ins(self.W, row), ins(self.mW, z), ins(self.vW, z.clone())
        self.b = ins(self.b, torch.full((1,), float(b), **f))
        self.a = ins(self.a, torch.full((1,), float(a), **f))
        self.mb, self.vb = ins(self.mb, s), ins(self.vb, s.clone())
        self.ma, self.va = ins(self.ma, s.clone()), ins(self.va, s.clone())
        self.t = ins(self.t, torch.zeros(1, dtype=torch.long, device=self.device))
        self.born = ins(self.born, torch.full((1,), int(born), dtype=torch.long,
                                              device=self.device))
        self.probe = ins(self.probe, torch.full((1,), bool(probe), dtype=torch.bool,
                                                device=self.device))

    def _keep_rows(self, keep: torch.Tensor) -> None:
        for name in ("W", "mW", "vW", "b", "mb", "vb", "a", "ma", "va", "t", "born", "probe"):
            setattr(self, name, getattr(self, name)[keep])

    def add_or(self, k, w, b, a, probe, born):
        at = self.ptr[k + 1]
        self._insert_rows(at, w, b, a, probe, born)
        for q in range(k + 1, len(self.ptr)):
            self.ptr[q] += 1
        self._rows_changed()

    def drop_or(self, k, r):
        i = self.ptr[k] + r
        keep = torch.ones(self.total_ors(), dtype=torch.bool, device=self.device)
        keep[i] = False
        self._keep_rows(keep)
        for q in range(k + 1, len(self.ptr)):
            self.ptr[q] -= 1
        self._rows_changed()

    # ------------------------------------------------------------- units

    def unit_probe(self, k):
        return self.uprobe[k]

    def set_unit_probe(self, k, v):
        self.uprobe[k] = bool(v)

    def unit_born(self, k):
        return self.uborn[k]

    def set_unit_born(self, k, v):
        self.uborn[k] = int(v)

    def add_unit(self):
        self.ptr.append(self.ptr[-1])
        self.uprobe.append(False)
        self.uborn.append(0)
        self._rows_changed()
        return self.n_out - 1

    def drop_unit(self, k):
        lo, hi = self.ptr[k], self.ptr[k + 1]
        keep = torch.ones(self.total_ors(), dtype=torch.bool, device=self.device)
        keep[lo:hi] = False
        self._keep_rows(keep)
        n = hi - lo
        del self.ptr[k + 1]
        for q in range(k + 1, len(self.ptr)):
            self.ptr[q] -= n
        del self.uprobe[k]
        del self.uborn[k]
        self._rows_changed()

    # ----------------------------------------------------------- columns

    def add_input(self):
        z = torch.zeros(self.total_ors(), 1, device=self.device, dtype=self.dtype)
        self.W = torch.cat([self.W, z], 1)
        self.mW = torch.cat([self.mW, z.clone()], 1)
        self.vW = torch.cat([self.vW, z.clone()], 1)
        self._n_in += 1
        self._rows_changed()
        return self._n_in - 1

    def drop_input(self, j):
        keep = torch.ones(self.n_in, dtype=torch.bool, device=self.device)
        keep[j] = False
        self.W, self.mW, self.vW = self.W[:, keep], self.mW[:, keep], self.vW[:, keep]
        self._n_in -= 1
        self._rows_changed()

    def scale_column(self, j, alpha, beta):
        self.b += self.W[:, j] * beta
        self.W[:, j] *= alpha
        self.mW[:, j] /= alpha
        self.vW[:, j] /= alpha * alpha

    def cols_get(self, js):
        if not len(js):
            return torch.zeros(self.total_ors(), 0, device=self.device, dtype=self.dtype)
        return self.W[:, js].clone()

    def cols_set(self, js, values):
        if len(js):
            self.W[:, js] = values.to(self.W)

    # ----------------------------------------------------------- compute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B, n_in) -> z (B, n_out); caches x and the Ors for backward."""
        B, m = x.shape[0], self.n_out
        o = x @ self.W.T + self.b                                   # (B, R)
        self._x, self._o = x, o
        if not self.total_ors():
            return torch.full((B, m), math.log(2.0), device=x.device, dtype=x.dtype)
        uid = self.uid().unsqueeze(0).expand(B, -1)
        nz = o != 0
        logabs = torch.where(nz, o.abs().log(), torch.zeros_like(o))
        ell = torch.zeros(B, m, device=x.device, dtype=x.dtype).scatter_add_(
            1, uid, self.a * logabs)
        nneg = torch.zeros_like(ell).scatter_add_(1, uid, (o < 0).to(o.dtype))
        nzero = torch.zeros_like(ell).scatter_add_(1, uid, (~nz).to(o.dtype))
        sgn = torch.where(nneg.remainder(2) == 0, 1.0, -1.0).to(o.dtype)
        z = sgn * torch.logaddexp(ell, torch.zeros_like(ell))
        return torch.where(nzero > 0, torch.zeros_like(z), z)

    def backward_adam(self, gz: torch.Tensor, lr: float, col_lr: torch.Tensor,
                      training: bool) -> torch.Tensor:
        """One sample (the last forward): dx with the pre-update weights, then
        one Adam step per Or (type_nn.c tnn_layer_backward_scaled)."""
        x, o = self._x[0], self._o[0]
        R, m = self.total_ors(), self.n_out
        if not R:
            return torch.zeros(self.n_in, device=self.device, dtype=self.dtype)
        uid = self.uid()
        nz = o != 0
        logabs = torch.where(nz, o.abs().log(), torch.zeros_like(o))
        alog = self.a * logabs
        ell = torch.zeros(m, device=o.device, dtype=o.dtype).scatter_add_(0, uid, alog)
        nzero = torch.zeros(m, device=o.device, dtype=o.dtype).scatter_add_(
            0, uid, (~nz).to(o.dtype))
        nneg = torch.zeros(m, device=o.device, dtype=o.dtype).scatter_add_(
            0, uid, (o < 0).to(o.dtype))
        sgn_nz = torch.where(nneg.remainder(2) == 0, 1.0, -1.0).to(o.dtype)
        sgn = torch.where(nzero > 0, torch.zeros_like(sgn_nz), sgn_nz)
        ell_full = torch.where(nzero > 0, torch.full_like(ell, -math.inf), ell)
        ss = (sgn * torch.sigmoid(ell_full))[uid]
        safe = torch.where(nz, o, torch.ones_like(o))
        dz_do = torch.where(nz, ss * self.a / safe, torch.zeros_like(o))
        dz_da = torch.where(nz, ss * logabs, torch.zeros_like(o))
        if bool((~nz).any()):
            # an exact zero factor: the cofactor when it is the unit's only zero
            # and a = 1, else 0 (C: zero_factor_slope; ell/sgn over the others)
            slope = sgn_nz[uid] * torch.exp(ell[uid])
            use = (~nz) & (nzero[uid] == 1) & (self.a <= 1.0)
            dz_do = torch.where(use, slope, dz_do)
        g = gz[uid]
        delta = g * dz_do
        ga = g * dz_da
        dx = delta @ self.W
        self.t += 1
        t = self.t.to(o.dtype)
        c1 = 1.0 / (1.0 - B1 ** t)
        c2 = 1.0 / (1.0 - B2 ** t)
        gw = delta.unsqueeze(1) * x.unsqueeze(0)
        self.mW.mul_(B1).add_(gw, alpha=1 - B1)
        self.vW.mul_(B2).addcmul_(gw, gw, value=1 - B2)
        self.W -= (lr * col_lr.unsqueeze(0)) * (self.mW * c1.unsqueeze(1)) / (
            (self.vW * c2.unsqueeze(1)).sqrt() + EPS)
        for p, mm, vv, gg in ((self.b, self.mb, self.vb, delta), (self.a, self.ma, self.va, ga)):
            mm.mul_(B1).add_(gg, alpha=1 - B1)
            vv.mul_(B2).addcmul_(gg, gg, value=1 - B2)
            p -= lr * (mm * c1) / ((vv * c2).sqrt() + EPS)
        self.a.clamp_(min=1.0)
        return dx

    # -------------------------------------------------------------- misc

    def clone(self) -> "PackedStore":
        c = PackedStore.__new__(PackedStore)
        c.__dict__.update(self.__dict__)
        for name in ("W", "mW", "vW", "b", "mb", "vb", "a", "ma", "va", "t", "born", "probe"):
            setattr(c, name, getattr(self, name).clone())
        c.ptr, c.uprobe, c.uborn = list(self.ptr), list(self.uprobe), list(self.uborn)
        c._uid = None
        c._x = c._o = None
        return c

    def load_or_state(self, k, r, o: dict) -> None:
        i = self.ptr[k] + r
        self.mW[i] = o["mw"].to(self.mW)
        self.vW[i] = o["vw"].to(self.vW)
        self.mb[i], self.vb[i] = o["mb"], o["vb"]
        self.ma[i], self.va[i] = o["ma"], o["va"]
        self.t[i] = o["t"]


class _PackedCore(GraphCore):
    def new_store(self, n_in, n_out):
        return PackedStore(n_in, n_out, self.device, self.dtype)


@dataclass
class CudaGraphFitResult:
    model: "CudaGraphTypeNN"
    counters: dict
    history: list = field(default_factory=list)


class CudaGraphTypeNN:
    """graph-type-nn with packed nodes on a torch device. Board row
    **cuda graph-type-nn**. The protocol is bench.c's: ``begin(n, epochs,
    lr)`` with the task lr (x 0.1 inside, as C), ``epoch(X, Y, seed)`` per
    epoch (xorshift shuffle + per-sample steps + epoch_end), ``end()``."""

    name = "cuda-graph"

    def __init__(self, in_features: int, out_features: int, *, seed: int = 1,
                 device=None, dtype=torch.float64) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.seed = int(seed)
        self.core = _PackedCore(in_features, out_features, seed, device=self.device,
                                dtype=dtype)
        self.core.t_sum = self.core.t_sum.to(self.device)
        self.core.t_sq = self.core.t_sq.to(self.device)
        self.core.build()
        self._begun = False

    # accounting
    @property
    def depth(self) -> int:
        return len(self.core.nodes)

    @property
    def birth_depth(self) -> int:
        return self.core.birth_vertices

    @property
    def n_edges(self) -> int:
        return len(self.core.edges)

    def num_params(self) -> int:
        return self.core.num_params()

    def structure(self) -> dict:
        return self.core.structure()

    def counters(self) -> dict:
        return self.core.counters.as_dict()

    def check(self):
        return self.core.check()

    # protocol
    def begin(self, n_train: int, epochs: int, lr: float) -> None:
        self.core.begin(n_train, epochs, float(lr) * LR_SCALE)
        self._begun = True

    def end(self) -> None:
        if self._begun:
            self.core.training = False
            self.core.end()

    def epoch_end(self) -> None:
        self.core.training = False
        self.core.epoch_end()

    def train(self, mode: bool = True) -> "CudaGraphTypeNN":
        self.core.training = bool(mode)
        return self

    def eval(self) -> "CudaGraphTypeNN":
        return self.train(False)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        t = _dev(x, self.device, self.core.dtype)
        squeeze = t.ndim == 1
        if squeeze:
            t = t.unsqueeze(0)
        tr, self.core.training = self.core.training, False
        y = self.core.forward(t)
        self.core.training = tr
        return y[0] if squeeze else y

    __call__ = forward

    @torch.no_grad()
    def step(self, x, t) -> torch.Tensor:
        """One bench.c sample: forward, mean-MSE backward, per-Or Adam."""
        return self.core.step_sample(_dev(x, self.device, self.core.dtype),
                                     _dev(t, self.device, self.core.dtype))

    @torch.no_grad()
    def epoch(self, X, Y, shuffle_seed: int) -> None:
        X = _dev(X, self.device, self.core.dtype)
        Y = _dev(Y, self.device, self.core.dtype)
        n = X.shape[0]
        order = list(range(n))
        s = shuffle_seed & MASK
        for i in range(n, 1, -1):
            s = xorshift32(s)
            j = s % i
            order[i - 1], order[j] = order[j], order[i - 1]
        self.core.training = True
        for i in order:
            self.core.step_sample(X[i], Y[i])
        self.core.training = False
        self.core.epoch_end()

    def fit(self, X, Y, *, epochs: int, lr: float, shuffle_seed=None) -> CudaGraphFitResult:
        """bench.c's loop; ``shuffle_seed(ep)`` gives the per-epoch seed
        (default ``34972 ^ ((ep + 1) * 0x9E3779B9)``, as bench.c)."""
        n = X.shape[0]
        self.begin(n, epochs, lr)
        hist = []
        for ep in range(epochs):
            seed = (shuffle_seed(ep) if shuffle_seed else
                    34972 ^ (((ep + 1) * 0x9E3779B9) & MASK))
            self.epoch(X, Y, seed)
            hist.append({"epoch": ep, "params": self.num_params(), "nodes": self.depth,
                         "edges": self.n_edges})
        self.end()
        return CudaGraphFitResult(self, self.counters(), hist)

    @classmethod
    def from_snapshot(cls, snap: dict, *, device=None, dtype=torch.float64):
        from ..graph.torch_store import load_snapshot
        m = cls(snap["n_in"], snap["n_out"], seed=1, device=device, dtype=dtype)
        load_snapshot(m.core, snap)
        m._begun = True
        return m


def _dev(x, device, dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)
