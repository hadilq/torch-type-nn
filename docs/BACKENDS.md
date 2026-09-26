# Backends

| backend | class | layout | board rows |
|---|---|---|---|
| `torch` | `TypeNN` | padded `(m,R,n)` + mask | torch type-nn* |
| `c` | `NativeTypeNN` | one `w[n_in]` per live Or | **C type-nn**, **C type-nn-overfit** |
| `cuda` | reserved | same ragged layout on device | — |

```python
from torch_type_nn import NativeTypeNN, get_backend
net = NativeTypeNN(4, 3, rule="threshold")   # C type-nn-overfit
get_backend("c")                             # NativeTypeNN
```

C sources sit in `src/torch_type_nn/native/c` and ship as package data
(no hatch `force-include`, which packed `ORIGIN` twice). Override the
type-nn snapshot with `TYPE_NN_SRC`.
