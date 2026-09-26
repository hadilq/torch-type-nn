"""Python handle for the C type-nn and type-nn-overfit models."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import _lib

__all__ = ["NativeTypeNN", "NativeFitResult", "native_available"]

RULES = {
    "bic": _lib.TNN_PY_BIC,
    "type-nn": _lib.TNN_PY_BIC,
    "threshold": _lib.TNN_PY_THRESHOLD,
    "overfit": _lib.TNN_PY_THRESHOLD,
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
    """The C network: ragged Ors, per-sample Adam, C's own scaler.

    Args:
        in_features, out_features: task widths.
        rule: ``"threshold"`` (C ``type-nn-overfit``) or ``"bic"`` (C ``type-nn``).
        seed: C's xorshift32 seed.
        backend: must be ``"c"``. ``"cuda"`` is reserved and rejected here
            so a later device store can take the same class name in another
            module without colliding.

    Memory: each live Or owns ``n_in`` weights. There is no padded ``R``
    and no ``mask``. Forward is one sample at a time — that is the C
    protocol, not a limitation of this wrapper.

    Training is C's loop: ``begin`` plants probes, each ``step`` is
    forward + mean-MSE gradient + one Adam update, ``epoch_end`` grows
    or prunes, ``end`` drops leftover probes. ``lr`` is the *task* rate;
    C multiplies it by ``0.1`` internally (the same scale the board uses).
    """

    name = "c"

    def __init__(self, in_features: int, out_features: int, *,
                 rule: str = "threshold", seed: int = 1,
                 backend: str = "c") -> None:
        if backend != "c":
            raise ValueError(
                f"NativeTypeNN is the C backend; got backend={backend!r}. "
                "Use torch_type_nn.get_backend('cuda') when that store exists."
            )
        if rule not in RULES:
            raise ValueError(f"rule must be one of {sorted(RULES)}, not {rule!r}")
        self.rule_name = "threshold" if RULES[rule] == _lib.TNN_PY_THRESHOLD else "bic"
        self._rule = RULES[rule]
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.seed = int(seed)
        self._lib = _lib.load()
        self._ptr = self._lib.tnn_py_create(
            self._rule, self.in_features, self.out_features, ctypes_uint(seed)
        )
        if not self._ptr:
            raise _lib.NativeUnavailable("tnn_py_create returned NULL")
        self._begun = False

    def _require_begun(self, what: str) -> None:
        if not self._begun:
            raise RuntimeError(
                f"{what} needs begin(n_train, epochs, lr) first "
                "(C builds the birth layers there, not in create)"
            )

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        lib = getattr(self, "_lib", None)
        rule = getattr(self, "_rule", None)
        if ptr and lib is not None:
            lib.tnn_py_free(ptr, rule)
            self._ptr = None

    def describe(self) -> str:
        return (f"C {self.rule_name}: ragged Ors, per-sample, "
                f"{self.num_params()} live parameters, depth {self.depth}")

    # ---------------------------------------------------------------- shapes

    @property
    def depth(self) -> int:
        return int(self._lib.tnn_py_depth(self._ptr, self._rule))

    @property
    def birth_depth(self) -> int:
        return int(self._lib.tnn_py_init_depth(self._ptr, self._rule))

    def num_params(self) -> int:
        return int(self._lib.tnn_py_params(self._ptr, self._rule))

    def structure(self) -> list[list[int]]:
        """Live Ors per unit, per layer (no padding)."""
        import ctypes
        cap = 4096
        buf = (ctypes.c_int * cap)()
        n_layers = ctypes.c_int()
        widths = (ctypes.c_int * 256)()
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

    # ------------------------------------------------------- training protocol

    def begin(self, n_train: int, epochs: int, lr: float) -> None:
        self._lib.tnn_py_begin(self._ptr, self._rule, int(n_train), int(epochs), float(lr))
        self._begun = True

    def end(self) -> None:
        if self._begun:
            self._lib.tnn_py_end(self._ptr, self._rule)

    def epoch_end(self) -> None:
        self._lib.tnn_py_epoch_end(self._ptr, self._rule)

    def train(self, mode: bool = True) -> "NativeTypeNN":
        self._lib.tnn_py_set_training(self._ptr, self._rule, 1 if mode else 0)
        return self

    def eval(self) -> "NativeTypeNN":
        return self.train(False)

    def forward(self, x) -> torch.Tensor:
        """``x`` shape ``(n,)`` or ``(B, n)``. C runs per row; output matches."""
        self._require_begun("forward")
        import ctypes
        t = _cpu64(x)
        if t.ndim == 1:
            t = t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if t.shape[-1] != self.in_features:
            raise ValueError(f"expected n_in={self.in_features}, got {tuple(t.shape)}")
        y = torch.empty(t.shape[0], self.out_features, dtype=torch.float64)
        xp = ctypes.POINTER(ctypes.c_double)
        for i in range(t.shape[0]):
            self._lib.tnn_py_forward(
                self._ptr, self._rule,
                ctypes.cast(t[i].data_ptr(), xp),
                ctypes.cast(y[i].data_ptr(), xp),
            )
        return y[0] if squeeze else y

    __call__ = forward

    def step(self, x, t) -> torch.Tensor:
        """One C training step: forward, mean-MSE gradient, Adam.

        ``x`` and ``t`` are a single sample (``n,``) / (``m,``).
        """
        import ctypes
        y = self.forward(x)
        tgt = _cpu64(t).reshape(-1)
        pred = y.reshape(-1)
        dy = (pred - tgt) / self.out_features
        dy = dy.contiguous()
        xp = ctypes.POINTER(ctypes.c_double)
        self._lib.tnn_py_backward(self._ptr, self._rule,
                                  ctypes.cast(dy.data_ptr(), xp))
        return y

    def fit(self, X, Y, *, epochs: int, lr: float,
            shuffle: bool = True, generator: torch.Generator | None = None) -> NativeFitResult:
        """C's per-sample protocol on a stacked dataset ``(N, n)``, ``(N, m)``."""
        X = _cpu64(X)
        Y = _cpu64(Y)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("fit expects X (N, n) and Y (N, m)")
        n = X.shape[0]
        self.begin(n, epochs, lr)
        hist = []
        for ep in range(epochs):
            if shuffle:
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


def ctypes_uint(seed: int):
    import ctypes
    return ctypes.c_uint(seed & 0xFFFFFFFF)
