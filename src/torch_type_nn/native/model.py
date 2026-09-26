from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import _lib

__all__ = ["NativeTypeNN", "NativeFitResult", "native_available"]

RULES = {
    "bic": _lib.TNN_PY_BIC, "type-nn": _lib.TNN_PY_BIC,
    "threshold": _lib.TNN_PY_THRESHOLD, "overfit": _lib.TNN_PY_THRESHOLD,
    "type-nn-overfit": _lib.TNN_PY_THRESHOLD,
}


def native_available() -> tuple[bool, str]:
    return _lib.native_available()


@dataclass
class NativeFitResult:
    model: "NativeTypeNN"
    counters: dict
    history: list[dict] = field(default_factory=list)


class NativeTypeNN:
    """C type-nn / type-nn-overfit: one w[n_in] per live Or, no padded mask."""

    name = "c"

    def __init__(self, in_features: int, out_features: int, *,
                 rule: str = "threshold", seed: int = 1, backend: str = "c") -> None:
        if backend != "c":
            raise ValueError(f"NativeTypeNN is the C backend, not {backend!r}")
        if rule not in RULES:
            raise ValueError(f"rule must be one of {sorted(RULES)}, not {rule!r}")
        self.rule_name = "threshold" if RULES[rule] == _lib.TNN_PY_THRESHOLD else "bic"
        self._rule = RULES[rule]
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.seed = int(seed)
        self._lib = _lib.load()
        import ctypes
        self._ptr = self._lib.tnn_py_create(
            self._rule, self.in_features, self.out_features, ctypes.c_uint(seed & 0xFFFFFFFF))
        if not self._ptr:
            raise _lib.NativeUnavailable("tnn_py_create returned NULL")
        self._begun = False

    def __del__(self) -> None:
        ptr, lib, rule = (getattr(self, n, None) for n in ("_ptr", "_lib", "_rule"))
        if ptr and lib is not None:
            lib.tnn_py_free(ptr, rule)
            self._ptr = None

    def _need_begin(self, what: str) -> None:
        if not self._begun:
            raise RuntimeError(f"{what} needs begin(n_train, epochs, lr) first")

    @property
    def depth(self) -> int:
        return int(self._lib.tnn_py_depth(self._ptr, self._rule))

    @property
    def birth_depth(self) -> int:
        return int(self._lib.tnn_py_init_depth(self._ptr, self._rule))

    def num_params(self) -> int:
        return int(self._lib.tnn_py_params(self._ptr, self._rule))

    def structure(self) -> list[list[int]]:
        import ctypes
        cap, buf = 4096, (ctypes.c_int * 4096)()
        n_layers, widths = ctypes.c_int(), (ctypes.c_int * 256)()
        n = self._lib.tnn_py_structure(self._ptr, self._rule, buf, cap,
                                       ctypes.byref(n_layers), widths)
        if n < 0:
            raise RuntimeError("structure buffer too small")
        out, i = [], 0
        for layer in range(n_layers.value):
            w = widths[layer]
            out.append([int(buf[i + k]) for k in range(w)])
            i += w
        return out

    def counters(self) -> dict[str, int]:
        import ctypes
        raw = (ctypes.c_uint * 6)()
        self._lib.tnn_py_counters(self._ptr, self._rule, raw)
        names = ("or_add", "or_drop", "and_add", "and_drop", "layer_add", "layer_drop")
        return {k: int(raw[i]) for i, k in enumerate(names)}

    def begin(self, n_train: int, epochs: int, lr: float) -> None:
        self._lib.tnn_py_begin(self._ptr, self._rule, int(n_train), int(epochs), float(lr))
        self._begun = True

    def end(self) -> None:
        if self._begun:
            self._lib.tnn_py_end(self._ptr, self._rule)

    def epoch_end(self) -> None:
        self._lib.tnn_py_epoch_end(self._ptr, self._rule)

    def train(self, mode: bool = True) -> "NativeTypeNN":
        self._lib.tnn_py_set_training(self._ptr, self._rule, int(bool(mode)))
        return self

    def eval(self) -> "NativeTypeNN":
        return self.train(False)

    def forward(self, x) -> torch.Tensor:
        self._need_begin("forward")
        import ctypes
        t = _cpu64(x)
        squeeze = t.ndim == 1
        if squeeze:
            t = t.unsqueeze(0)
        if t.shape[-1] != self.in_features:
            raise ValueError(f"expected n_in={self.in_features}, got {tuple(t.shape)}")
        y = torch.empty(t.shape[0], self.out_features, dtype=torch.float64)
        xp = ctypes.POINTER(ctypes.c_double)
        for i in range(t.shape[0]):
            self._lib.tnn_py_forward(self._ptr, self._rule,
                                     ctypes.cast(t[i].data_ptr(), xp),
                                     ctypes.cast(y[i].data_ptr(), xp))
        return y[0] if squeeze else y

    __call__ = forward

    def step(self, x, t) -> torch.Tensor:
        """One bench.c sample: forward + mean-MSE backward, both in C."""
        import ctypes
        x64, t64 = _cpu64(x).reshape(-1), _cpu64(t).reshape(-1)
        if x64.numel() != self.in_features or t64.numel() != self.out_features:
            raise ValueError("step expected a single sample")
        xp = ctypes.POINTER(ctypes.c_double)
        self._lib.tnn_py_step(self._ptr, self._rule,
                              ctypes.cast(x64.data_ptr(), xp),
                              ctypes.cast(t64.data_ptr(), xp))
        return self.forward(x64)

    def epoch(self, X, Y, shuffle_seed: int) -> None:
        """One bench.c epoch: C xorshift shuffle, train, epoch_end."""
        import ctypes
        X64, Y64 = _cpu64(X), _cpu64(Y)
        if X64.ndim != 2:
            X64 = X64.reshape(1, -1)
            Y64 = Y64.reshape(1, -1)
        n = X64.shape[0]
        if X64.shape[1] != self.in_features or Y64.shape[1] != self.out_features:
            raise ValueError("epoch expected (n, n_in) and (n, n_out)")
        Xc, Yc = X64.contiguous(), Y64.contiguous()
        xp = ctypes.POINTER(ctypes.c_double)
        self._need_begin("epoch")
        self._lib.tnn_py_epoch(
            self._ptr, self._rule,
            ctypes.cast(Xc.data_ptr(), xp),
            ctypes.cast(Yc.data_ptr(), xp),
            n, ctypes.c_uint(shuffle_seed & 0xFFFFFFFF))

    def fit(self, X, Y, *, epochs: int, lr: float,
            shuffle=True, generator: torch.Generator | None = None) -> NativeFitResult:
        X, Y = _cpu64(X), _cpu64(Y)
        n = X.shape[0]
        self.begin(n, epochs, lr)
        hist = []
        for ep in range(epochs):
            if callable(shuffle):
                order = shuffle(ep, n)
            elif shuffle:
                order = torch.randperm(n, generator=generator)
            else:
                order = torch.arange(n)
            self.train(True)
            sq = 0.0
            for i in order.tolist():
                y = self.step(X[i], Y[i])
                sq += float(((y - Y[i]) ** 2).sum())
            self.train(False)
            self.epoch_end()
            hist.append({"epoch": ep, "train_mse": sq / (n * self.out_features),
                         "params": self.num_params(), "depth": self.depth})
        self.end()
        return NativeFitResult(self, self.counters(), hist)


def _cpu64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float64, device="cpu").contiguous()
    return torch.as_tensor(x, dtype=torch.float64).contiguous()
