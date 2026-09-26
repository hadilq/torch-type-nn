# Backends

type-nn has two jobs that pull the store in opposite directions.

1. **Learn the structure.** Width, degree and depth grow and shrink.
   Memory should follow the live Ors: one weight vector of length `n_in`
   per Or, nothing padded.
2. **Train in batch on an accelerator.** A GEMM wants a dense
   `(m, R, n)` tensor. Free slots become identity Ors (`w = 0, b = 1`)
   and a `mask`, which is extra memory.

`torch-type-nn` keeps both stores and lets you pick one. They do not
share weights.

| backend | class | layout | batch | scaler |
|---|---|---|---|---|
| `torch` (default) | `TypeNN` / `AndOr` | padded `(m, R, n)` + mask | yes | `StructureScaler` |
| `c` | `NativeTypeNN` | C's ragged Or list | one sample | C's own (`type_nn_scale.c` or `type_nn_overfit_scale.c`) |
| `cuda` | *reserved* | same ragged layout **on device** | later | later |

```python
from torch_type_nn import TypeNN, NativeTypeNN, get_backend, available_backends

torch_net = TypeNN(4, 3)                 # padded, batched
c_net     = NativeTypeNN(4, 3, rule="threshold")   # C type-nn-overfit
CTypeNN   = get_backend("c")             # same class
available_backends()
```

## C backend

`NativeTypeNN(..., rule=)` selects the C model:

- `"threshold"` / `"overfit"` — `type-nn-overfit` (back-prop + band)
- `"bic"` / `"type-nn"` — `type-nn` (BIC prior on training pairs)

The C sources vendored under `src/torch_type_nn/native/c/` are the board
snapshot `7333bf7`. A checkout pointed at by `TYPE_NN_SRC` is used
instead if it contains `type_nn.c`. The shared library is compiled on
first use into `$XDG_CACHE_HOME/torch-type-nn/` (needs `cc` or `gcc`).

`lr` is the *task* rate from the board. C multiplies it by `0.1`
inside `tnn_begin` / `tnno_begin`.

```python
result = c_net.fit(X, Y, epochs=250, lr=0.05)
c_net.structure()     # live Ors per unit; no padding
c_net.num_params()
c_net.counters()
```

A single step is C's protocol: forward, `dy = (y - t) / m`, Adam.

## CUDA (not shipped)

`get_backend("cuda")` raises `NotImplementedError` on purpose. The
intended store is **not** a port of the padded torch tensors. It is
the C layout on device:

- per layer: `n_out` units, each with `n_or` Ors
- per Or: `w[n_in]`, `b`, `a`, Adam moments
- no `R` pad, no identity slots
- a compact kernel that gathers the live Ors of a batch, or a
  per-sample kernel that matches C while the structure is small

When that exists it should implement the same surface as
`NativeTypeNN` (`forward`, `step`, `begin` / `epoch_end` / `end`,
`structure`, `num_params`) and register under `get_backend("cuda")`.
