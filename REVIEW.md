# torch-type-nn — C / CUDA backend snapshot

Adds a Python wrapper around the C type-nn models so memory can follow
the live Ors instead of a padded torch mask.

## New

- `NativeTypeNN(n, m, rule="threshold"|"bic")` — ctypes + vendored C
  (`src/torch_type_nn/native/c/`, snapshot `7333bf7`)
- `get_backend("torch"|"c"|"cuda")`, `available_backends()`
- `docs/BACKENDS.md`

## Run

```sh
python -m venv .venv && . .venv/bin/activate
pip install torch pytest
export PYTHONPATH=src
pytest tests/test_native.py tests/test_package.py -q
```

```python
from torch_type_nn import NativeTypeNN, get_backend
net = NativeTypeNN(2, 1, rule="threshold", seed=1)
net.fit(X, Y, epochs=400, lr=0.08)
print(net.structure(), net.num_params())
```
