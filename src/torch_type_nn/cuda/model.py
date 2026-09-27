"""Ragged type-nn store on a torch device (CUDA when present).

Each live Or owns ``w[n_in]``. A layer packs those rows into ``W (R, n_in)``
so forward is one GEMM (``X @ W.T + b``) and the And is a segmented product,
not a Python loop over Ors and not a padded ``(m, R, n)`` mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..network import birth_depth

__all__ = ["CudaTypeNN", "CudaFitResult", "cuda_available"]

MASK = 0xFFFFFFFF
B1, B2, EPS = 0.9, 0.999, 1e-8
LR_SCALE = 0.1
GROW, FIT, PRUNE, DONE = 0, 1, 2, 3


def cuda_available() -> tuple[bool, str]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        return True, f"{name} cap {cap[0]}.{cap[1]} (torch {torch.__version__})"
    return False, "no CUDA device (ragged store still runs on cpu)"


def _xorshift32(s: int) -> int:
    x = s if s else 2463534242
    x ^= (x << 13) & MASK
    x ^= x >> 17
    x ^= (x << 5) & MASK
    return x & MASK


def _uniform(rng: list[int]) -> float:
    rng[0] = _xorshift32(rng[0])
    return (rng[0] / 4294967296.0) * 2.0 - 1.0


def _phase(u: float) -> int:
    if u < 1.0 / 3.0:
        return GROW
    if u < 2.0 / 3.0:
        return FIT
    return PRUNE


def _dev64(x, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float64, device=device).contiguous()
    return torch.as_tensor(x, dtype=torch.float64, device=device).contiguous()


def _draw_or(n_in: int, rng: list[int], *, identity: bool, noise: float):
    w = torch.empty(n_in, dtype=torch.float64)
    if identity:
        for j in range(n_in):
            w[j] = noise * _uniform(rng)
        b = 1.0 + noise * _uniform(rng)
    else:
        k = 1.0 / math.sqrt(max(n_in, 1))
        for j in range(n_in):
            w[j] = k * _uniform(rng)
        b = k * _uniform(rng)
    return w, b, 1.0


@dataclass
class PackedLayer:
    """One layer: ``R`` Or rows packed for a fused matmul."""

    n_in: int
    n_out: int
    W: torch.Tensor
    mW: torch.Tensor
    vW: torch.Tensor
    b: torch.Tensor
    mb: torch.Tensor
    vb: torch.Tensor
    a: torch.Tensor
    ma: torch.Tensor
    va: torch.Tensor
    t: torch.Tensor
    born: torch.Tensor
    probe: torch.Tensor
    ptr: torch.Tensor
    x: torch.Tensor | None = None
    gin: float = 0.0

    @property
    def n_or(self) -> int:
        return int(self.W.shape[0])

    def unit_id(self) -> torch.Tensor:
        m = self.n_out
        counts = self.ptr[1:] - self.ptr[:-1]
        return torch.repeat_interleave(torch.arange(m, device=self.W.device), counts)

    def num_ors(self) -> list[int]:
        p = self.ptr.tolist()
        return [p[k + 1] - p[k] for k in range(self.n_out)]

    def num_params(self) -> int:
        return self.n_or * (self.n_in + 2)

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(B, n_in)``; returns ``(B, n_out)``."""
        self.x = x
        o = x @ self.W.T + self.b          # (B, R)
        uid = self.unit_id().unsqueeze(0).expand(x.shape[0], -1)
        logabs = torch.where(o != 0, o.abs().log(), torch.zeros_like(o))
        ell = torch.zeros(x.shape[0], self.n_out, dtype=x.dtype, device=x.device)
        ell.scatter_add_(1, uid, self.a * logabs)
        nneg = torch.zeros_like(ell)
        nzero = torch.zeros_like(ell)
        nneg.scatter_add_(1, uid, (o < 0).to(o.dtype))
        nzero.scatter_add_(1, uid, (o == 0).to(o.dtype))
        sgn = torch.where(nzero > 0, torch.zeros_like(ell),
                          torch.where(nneg.to(torch.int64) % 2 == 0,
                                      torch.ones_like(ell), -torch.ones_like(ell)))
        z = sgn * torch.logaddexp(ell, torch.zeros_like(ell))
        return torch.where(nzero > 0, torch.zeros_like(z), z)

    def backward_one(self, gz: torch.Tensor, lr: float) -> torch.Tensor:
        """One-sample backward + Adam. ``gz`` is ``(n_out,)``."""
        x = self.x
        assert x is not None and x.shape[0] == 1
        o = x @ self.W.T + self.b          # (1, R)
        uid = self.unit_id()
        o1 = o[0]
        logabs = torch.where(o1 != 0, o1.abs().log(), torch.zeros_like(o1))
        ell = torch.zeros(self.n_out, dtype=o.dtype, device=o.device)
        ell.scatter_add_(0, uid, self.a * logabs)
        nneg = torch.zeros_like(ell)
        nzero = torch.zeros_like(ell)
        nneg.scatter_add_(0, uid, (o1 < 0).to(o.dtype))
        nzero.scatter_add_(0, uid, (o1 == 0).to(o.dtype))
        sgn = torch.where(nzero > 0, torch.zeros_like(ell),
                          torch.where(nneg.to(torch.int64) % 2 == 0,
                                      torch.ones_like(ell), -torch.ones_like(ell)))
        ss = sgn * torch.sigmoid(ell)
        ss_r = ss[uid]
        nz = o1 != 0
        safe = torch.where(nz, o1, torch.ones_like(o1))
        dz_do = torch.where(nz, ss_r * self.a / safe, torch.zeros_like(o1))
        dz_da = torch.where(nz, ss_r * logabs, torch.zeros_like(o1))
        delta = gz[uid] * dz_do
        ga = gz[uid] * dz_da
        dx = delta @ self.W
        self.t = self.t + 1
        gw = delta.unsqueeze(-1) * x[0]
        self.W = self.W - _adam(self.mW, self.vW, gw, lr, self.t)
        self.b = self.b - _adam(self.mb, self.vb, delta, lr, self.t)
        self.a = self.a - _adam(self.ma, self.va, ga, lr, self.t)
        self.a = torch.clamp(self.a, min=1.0)
        return dx


def _adam(m: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
          lr: float, t: torch.Tensor) -> torch.Tensor:
    m.mul_(B1).add_(g, alpha=1.0 - B1)
    v.mul_(B2).addcmul_(g, g, value=1.0 - B2)
    # t can differ across rows (later-born probes)
    shape = [1] * g.ndim
    shape[0] = -1
    tt = t.reshape(shape).to(dtype=g.dtype)
    c1 = 1.0 / (1.0 - B1 ** tt)
    c2 = 1.0 / (1.0 - B2 ** tt)
    return lr * (m * c1) / (v.mul(c2).sqrt() + EPS)


def pack_layer(n_in: int, rows: list[list[tuple]], device) -> PackedLayer:
    """``rows[k]`` is a list of ``(w, b, a, t, born, probe)`` for unit k."""
    W, mW, vW, b, mb, vb, a, ma, va, t, born, probe = (
        [], [], [], [], [], [], [], [], [], [], [], [])
    ptr = [0]
    for unit in rows:
        for w, bb, aa, tt, br, pr in unit:
            W.append(w.to(device))
            z = torch.zeros_like(w, device=device)
            mW.append(z)
            vW.append(z.clone())
            b.append(torch.as_tensor(bb, dtype=torch.float64, device=device))
            s = torch.zeros((), dtype=torch.float64, device=device)
            mb.append(s)
            vb.append(s.clone())
            a.append(torch.as_tensor(aa, dtype=torch.float64, device=device))
            ma.append(s.clone())
            va.append(s.clone())
            t.append(int(tt))
            born.append(int(br))
            probe.append(int(pr))
        ptr.append(ptr[-1] + len(unit))
    return PackedLayer(
        n_in=n_in, n_out=len(rows),
        W=torch.stack(W), mW=torch.stack(mW), vW=torch.stack(vW),
        b=torch.stack(b), mb=torch.stack(mb), vb=torch.stack(vb),
        a=torch.stack(a), ma=torch.stack(ma), va=torch.stack(va),
        t=torch.tensor(t, dtype=torch.int64, device=device),
        born=torch.tensor(born, dtype=torch.int64, device=device),
        probe=torch.tensor(probe, dtype=torch.int64, device=device),
        ptr=torch.tensor(ptr, dtype=torch.int64, device=device),
    )


def _dev_rows(layer: PackedLayer) -> torch.Tensor:
    s = (layer.W * layer.W).sum(-1)
    s = s + (layer.b - 1.0) ** 2 + (layer.a - 1.0) ** 2
    return torch.sqrt(s / (layer.n_in + 2))


@dataclass
class CudaFitResult:
    model: "CudaTypeNN"
    counters: dict
    history: list[dict] = field(default_factory=list)


class CudaTypeNN:
    """Ragged And-of-Ors packed for fused device matmuls."""

    name = "cuda"

    def __init__(self, in_features: int, out_features: int, *,
                 rule: str = "threshold", seed: int = 1,
                 device=None, backend: str = "cuda") -> None:
        if backend not in ("cuda", "gpu"):
            raise ValueError(f"CudaTypeNN is the cuda backend, not {backend!r}")
        if rule not in ("threshold", "overfit", "type-nn-overfit", "bic", "type-nn"):
            raise ValueError(f"unknown rule {rule!r}")
        self.rule_name = "bic" if rule in ("bic", "type-nn") else "threshold"
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.seed = int(seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.layers: list[PackedLayer] = []
        self._rng = [self.seed if self.seed else 1]
        self._begun = False
        self.lr = 0.0
        self.n_train = 0
        self.epochs = 0
        self.step_i = 0
        self.total = 1
        self.phase = GROW
        self.init_depth = 0
        self.training_mode = False
        self.or_add = self.or_drop = 0
        self.and_add = self.and_drop = 0
        self.layer_add = self.layer_drop = 0
        self._cx: torch.Tensor | None = None
        self._ct: torch.Tensor | None = None
        self.epoch_loss = 0.0
        self._loss_n = 0

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def birth_depth(self) -> int:
        return self.init_depth

    def num_params(self) -> int:
        return sum(layer.num_params() for layer in self.layers)

    def structure(self) -> list[list[int]]:
        return [layer.num_ors() for layer in self.layers]

    def counters(self) -> dict[str, int]:
        return {
            "or_add": self.or_add, "or_drop": self.or_drop,
            "and_add": self.and_add, "and_drop": self.and_drop,
            "layer_add": self.layer_add, "layer_drop": self.layer_drop,
        }

    def begin(self, n_train: int, epochs: int, lr: float) -> None:
        self.n_train = int(n_train)
        self.epochs = int(epochs)
        self.lr = float(lr) * LR_SCALE
        self.step_i = 0
        self.total = max(1, self.n_train * self.epochs)
        self.phase = GROW
        self.layers = []
        d = birth_depth(self.in_features, self.out_features)
        self.init_depth = d
        n = self.in_features
        for _ in range(d):
            units = []
            for _k in range(self.out_features):
                w, b, a = _draw_or(n, self._rng, identity=False, noise=0.0)
                units.append([(w, b, a, 0, 0, 0)])
            self.layers.append(pack_layer(n, units, self.device))
            n = self.out_features
        self._grow_degree()
        self._begun = True

    def end(self) -> None:
        if self._begun:
            self._prune()
            self.phase = DONE

    def epoch_end(self) -> None:
        u = self.step_i / self.total
        ph = _phase(u)
        if self.phase == DONE:
            return
        if ph == GROW and self._residual():
            self._grow_degree()
        elif ph == PRUNE:
            self._prune()
        self.phase = ph
        self._cx = self._ct = None
        self.epoch_loss = 0.0
        self._loss_n = 0
        for layer in self.layers:
            layer.gin = 0.0

    def train(self, mode: bool = True) -> "CudaTypeNN":
        self.training_mode = bool(mode)
        return self

    def eval(self) -> "CudaTypeNN":
        return self.train(False)

    def _need_begin(self, what: str) -> None:
        if not self._begun:
            raise RuntimeError(f"{what} needs begin(n_train, epochs, lr) first")

    def forward(self, x) -> torch.Tensor:
        self._need_begin("forward")
        t = _dev64(x, self.device)
        squeeze = t.ndim == 1
        if squeeze:
            t = t.unsqueeze(0)
        if t.shape[-1] != self.in_features:
            raise ValueError(f"expected n_in={self.in_features}, got {tuple(t.shape)}")
        y = t
        for layer in self.layers:
            y = layer.forward_batch(y)
        return y[0] if squeeze else y

    __call__ = forward

    def step(self, x, t) -> torch.Tensor:
        self._need_begin("step")
        x64 = _dev64(x, self.device).reshape(1, -1)
        t64 = _dev64(t, self.device).reshape(-1)
        y = x64
        for layer in self.layers:
            y = layer.forward_batch(y)
        y = y[0]
        dy = (y - t64) / self.out_features
        gz = dy
        for layer in reversed(self.layers):
            dx = layer.backward_one(gz, self.lr)
            gz = dx
            if self.training_mode and layer.n_in:
                layer.gin += float(dx.abs().mean())
        if self.training_mode:
            self._push_cache(x64[0], t64)
            self.epoch_loss += float(((y - t64) ** 2).mean())
            self._loss_n += 1
            self.step_i += 1
        return y

    def _push_cache(self, x: torch.Tensor, t: torch.Tensor) -> None:
        x = x.detach().unsqueeze(0)
        t = t.detach().unsqueeze(0)
        if self._cx is None:
            self._cx, self._ct = x.clone(), t.clone()
        else:
            self._cx = torch.cat([self._cx, x], 0)
            self._ct = torch.cat([self._ct, t], 0)

    def _residual(self) -> bool:
        if self._loss_n == 0:
            return True
        return (self.epoch_loss / self._loss_n) > 1e-3

    def _theta_up(self, age: torch.Tensor) -> torch.Tensor:
        return self.lr * age.clamp(min=1).to(torch.float64).pow(0.75)

    def _theta_band(self) -> float:
        return self.lr * (max(self.n_train, 1) ** 0.75)

    def _grow_degree(self) -> None:
        for i, layer in enumerate(self.layers):
            dev = _dev_rows(layer)
            age = self.step_i - layer.born
            promote = (layer.probe == 1) & (dev > self._theta_up(age))
            layer.probe = torch.where(promote, torch.zeros_like(layer.probe), layer.probe)
            self.and_add += int(promote.sum())
            self.layers[i] = self._append_probes(layer)

    def _unpack(self, layer: PackedLayer):
        rows = []
        p = layer.ptr.tolist()
        for k in range(layer.n_out):
            unit = []
            for r in range(p[k], p[k + 1]):
                unit.append((
                    layer.W[r].detach().cpu(),
                    float(layer.b[r]), float(layer.a[r]),
                    int(layer.t[r]), int(layer.born[r]), int(layer.probe[r]),
                ))
            rows.append(unit)
        return rows

    def _append_probes(self, layer: PackedLayer) -> PackedLayer:
        """Add one identity probe to units that lack one; keep Adam state."""
        p = layer.ptr.tolist()
        dev = layer.W.device
        chunks = {k: [] for k in (
            "W", "mW", "vW", "b", "mb", "vb", "a", "ma", "va", "t", "born", "probe")}
        ptr = [0]
        for k in range(layer.n_out):
            sl = slice(p[k], p[k + 1])
            chunks["W"].append(layer.W[sl])
            chunks["mW"].append(layer.mW[sl])
            chunks["vW"].append(layer.vW[sl])
            chunks["b"].append(layer.b[sl])
            chunks["mb"].append(layer.mb[sl])
            chunks["vb"].append(layer.vb[sl])
            chunks["a"].append(layer.a[sl])
            chunks["ma"].append(layer.ma[sl])
            chunks["va"].append(layer.va[sl])
            chunks["t"].append(layer.t[sl])
            chunks["born"].append(layer.born[sl])
            chunks["probe"].append(layer.probe[sl])
            extra = 0
            n_here = p[k + 1] - p[k]
            # First CUDA scaler: one probe per unit, no runaway degree.
            # Extra promotions stacked an And that blew up on xor (1,1).
            if n_here < 2 and not bool((layer.probe[sl] == 1).any()):
                w, b, a = _draw_or(layer.n_in, self._rng,
                                   identity=True, noise=self.lr)
                z = torch.zeros(layer.n_in, dtype=torch.float64, device=dev)
                s = torch.zeros((), dtype=torch.float64, device=dev)
                chunks["W"].append(w.to(dev).unsqueeze(0))
                chunks["mW"].append(z.unsqueeze(0))
                chunks["vW"].append(z.unsqueeze(0).clone())
                chunks["b"].append(torch.tensor([b], dtype=torch.float64, device=dev))
                chunks["mb"].append(s.unsqueeze(0))
                chunks["vb"].append(s.unsqueeze(0).clone())
                chunks["a"].append(torch.tensor([a], dtype=torch.float64, device=dev))
                chunks["ma"].append(s.unsqueeze(0).clone())
                chunks["va"].append(s.unsqueeze(0).clone())
                chunks["t"].append(torch.zeros(1, dtype=torch.int64, device=dev))
                chunks["born"].append(torch.tensor([self.step_i], dtype=torch.int64,
                                                   device=dev))
                chunks["probe"].append(torch.ones(1, dtype=torch.int64, device=dev))
                extra = 1
            ptr.append(ptr[-1] + (p[k + 1] - p[k]) + extra)
        return PackedLayer(
            n_in=layer.n_in, n_out=layer.n_out,
            W=torch.cat(chunks["W"]), mW=torch.cat(chunks["mW"]),
            vW=torch.cat(chunks["vW"]),
            b=torch.cat(chunks["b"]), mb=torch.cat(chunks["mb"]),
            vb=torch.cat(chunks["vb"]),
            a=torch.cat(chunks["a"]), ma=torch.cat(chunks["ma"]),
            va=torch.cat(chunks["va"]),
            t=torch.cat(chunks["t"]), born=torch.cat(chunks["born"]),
            probe=torch.cat(chunks["probe"]),
            ptr=torch.tensor(ptr, dtype=torch.int64, device=dev),
        )

    def _index_layer(self, layer: PackedLayer, keep: torch.Tensor,
                     ptr: list[int]) -> PackedLayer:
        """Compact live rows; keeps Adam moments."""
        dev = layer.W.device
        return PackedLayer(
            n_in=layer.n_in, n_out=layer.n_out,
            W=layer.W[keep], mW=layer.mW[keep], vW=layer.vW[keep],
            b=layer.b[keep], mb=layer.mb[keep], vb=layer.vb[keep],
            a=layer.a[keep], ma=layer.ma[keep], va=layer.va[keep],
            t=layer.t[keep], born=layer.born[keep],
            probe=torch.zeros(int(keep.sum()), dtype=torch.int64, device=dev),
            ptr=torch.tensor(ptr, dtype=torch.int64, device=dev),
        )

    def _prune(self) -> None:
        if self.rule_name == "bic" and self._cx is not None:
            self._prune_bic()
            return
        th = self._theta_band()
        for i, layer in enumerate(self.layers):
            dev = _dev_rows(layer)
            p = layer.ptr.tolist()
            mask = torch.zeros(layer.n_or, dtype=torch.bool, device=layer.W.device)
            ptr = [0]
            for k in range(layer.n_out):
                sl = slice(p[k], p[k + 1])
                n = p[k + 1] - p[k]
                if n <= 1:
                    mask[sl] = True
                    ptr.append(ptr[-1] + n)
                    continue
                d = dev[sl]
                best = int(torch.argmax(d))
                unit_keep = (d > th)
                unit_keep[best] = True
                dropped = int((~unit_keep).sum())
                self.and_drop += dropped
                mask[p[k]:p[k + 1]] = unit_keep
                ptr.append(ptr[-1] + int(unit_keep.sum()))
            if int(mask.sum()) == layer.n_or:
                layer.probe.zero_()
                continue
            self.layers[i] = self._index_layer(layer, mask, ptr)

    def _prune_bic(self) -> None:
        assert self._cx is not None
        n = self._cx.shape[0]
        nn = n * self.out_features
        base = self._cache_mse()
        price = math.log(max(nn, 2))

        def keep(mse_wo: float) -> bool:
            return n * math.log(max(mse_wo, 1e-18) / max(base, 1e-18)) > price

        for i, layer in enumerate(self.layers):
            rows = self._unpack(layer)
            p = layer.ptr.tolist()
            kept_all = []
            for k, unit in enumerate(rows):
                if len(unit) <= 1:
                    kept_all.append([(w, b, a, t, br, 0)
                                     for (w, b, a, t, br, _pr) in unit])
                    continue
                survivors = []
                for j, row in enumerate(unit):
                    r = p[k] + j
                    snap = (layer.W[r].clone(), float(layer.b[r]), float(layer.a[r]))
                    layer.W[r].zero_()
                    layer.b[r] = 1.0
                    layer.a[r] = 1.0
                    mse_wo = self._cache_mse()
                    layer.W[r].copy_(snap[0])
                    layer.b[r] = snap[1]
                    layer.a[r] = snap[2]
                    if keep(mse_wo) or row[5]:
                        w, b, a, t, br, _pr = row
                        survivors.append((w, b, a, t, br, 0))
                    elif row[5] == 0:
                        self.and_drop += 1
                kept_all.append(survivors or [unit[0][:5] + (0,)])
            self.layers[i] = pack_layer(layer.n_in, kept_all, self.device)

    def _cache_mse(self) -> float:
        assert self._cx is not None and self._ct is not None
        y = self._cx
        for layer in self.layers:
            y = layer.forward_batch(y)
        return float(((y - self._ct) ** 2).mean())

    def epoch(self, X, Y, shuffle_seed: int) -> None:
        X64, Y64 = _dev64(X, self.device), _dev64(Y, self.device)
        if X64.ndim != 2:
            X64, Y64 = X64.reshape(1, -1), Y64.reshape(1, -1)
        n = X64.shape[0]
        order = list(range(n))
        s = shuffle_seed & MASK
        for i in range(n, 1, -1):
            s = _xorshift32(s)
            j = s % i
            order[i - 1], order[j] = order[j], order[i - 1]
        self.train(True)
        for i in order:
            self.step(X64[i], Y64[i])
        self.train(False)
        self.epoch_end()

    def fit(self, X, Y, *, epochs: int, lr: float,
            shuffle: bool = True) -> CudaFitResult:
        X64, Y64 = _dev64(X, self.device), _dev64(Y, self.device)
        n = X64.shape[0]
        self.begin(n, epochs, lr)
        hist = []
        for ep in range(epochs):
            if shuffle:
                self.epoch(X64, Y64, (self.seed ^ ((ep + 1) * 0x9E3779B9)) & MASK)
            else:
                self.train(True)
                for i in range(n):
                    self.step(X64[i], Y64[i])
                self.train(False)
                self.epoch_end()
            hist.append({"epoch": ep, "params": self.num_params(),
                         "depth": self.depth})
        self.end()
        return CudaFitResult(self, self.counters(), hist)
