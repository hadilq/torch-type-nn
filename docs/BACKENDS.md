# Backends

| backend | class | layout | board rows |
|---|---|---|---|
| `torch` | `TypeNN` | padded `(m,R,n)` + mask | torch type-nn* |
| `c` | `NativeTypeNN` | one `w[n_in]` per live Or | **C type-nn**, **C type-nn-overfit** |
| `cuda` | `CudaTypeNN` | packed `W (R, n_in)` GEMM | cuda type-nn* |

```python
from torch_type_nn import CudaTypeNN, NativeTypeNN, get_backend
net = NativeTypeNN(4, 3, rule="threshold")   # C type-nn-overfit
gpu = CudaTypeNN(4, 3, rule="threshold")     # ragged store on cuda (or cpu)
get_backend("cuda")                          # CudaTypeNN
```

`CudaTypeNN` does not pad. Each layer packs live Ors into `W (R, n_in)`
and runs `X @ W.T + b` as one GEMM; the And is a segmented product.
On a machine with no GPU the same store runs on CPU so CI covers the
path. Structure scaling is the threshold rule; `rule="bic"` prunes on a
device-side pair cache.

```sh
python benchmarks/bench.py xor cuda-type-nn,cuda-type-nn-overfit --device cuda
```

C sources sit in `src/torch_type_nn/native/c` and ship as package data
(no hatch `force-include`, which packed `ORIGIN` twice). Override the
type-nn snapshot with `TYPE_NN_SRC`.
