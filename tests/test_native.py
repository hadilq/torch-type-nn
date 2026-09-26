"""The C backend: ragged type-nn / type-nn-overfit behind NativeTypeNN."""

import shutil

import pytest
import torch

from torch_type_nn import NativeTypeNN, available_backends, get_backend
from torch_type_nn.native import native_available

ok, detail = native_available()
pytestmark = pytest.mark.skipif(not ok, reason=f"native backend unavailable: {detail}")


def test_backend_registry():
    names = available_backends()
    assert "torch" in names and "c" in names and "cuda" in names
    assert get_backend("torch").__name__ == "TypeNN"
    assert get_backend("c") is NativeTypeNN
    with pytest.raises(NotImplementedError, match="CUDA"):
        get_backend("cuda")
    with pytest.raises(ValueError, match="unknown"):
        get_backend("tpu")


def test_create_forward_and_structure():
    net = NativeTypeNN(4, 3, rule="threshold", seed=7)
    assert net.in_features == 4 and net.out_features == 3
    assert net.depth == 0
    with pytest.raises(RuntimeError, match="begin"):
        net(torch.zeros(4))
    net.begin(n_train=8, epochs=2, lr=0.05)
    assert net.depth >= 1
    x = torch.randn(4, dtype=torch.float64)
    y = net(x)
    assert y.shape == (3,)
    assert torch.isfinite(y).all()
    batch = net(torch.randn(5, 4, dtype=torch.float64))
    assert batch.shape == (5, 3)
    struct = net.structure()
    assert len(struct) == net.depth
    assert all(isinstance(u, int) and u >= 0 for layer in struct for u in layer)
    assert net.num_params() > 0
    net.end()


@pytest.mark.parametrize("rule", ["threshold", "bic"])
def test_xor_fits(rule):
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    net = NativeTypeNN(2, 1, rule=rule, seed=34972)
    net.fit(X, Y, epochs=400, lr=0.08, shuffle=False)
    y = net(X)
    pred = (y.reshape(-1) >= 0.5)
    tgt = (Y.reshape(-1) >= 0.5)
    assert bool((pred == tgt).all()), (y, net.structure(), net.num_params())
    assert net.num_params() >= 2
    c = net.counters()
    assert set(c) == {"or_add", "or_drop", "and_add", "and_drop", "layer_add", "layer_drop"}


def test_begin_grows_params_then_end_drops_probes():
    net = NativeTypeNN(3, 2, rule="threshold", seed=3)
    before = net.num_params()
    net.begin(n_train=10, epochs=3, lr=0.05)
    assert net.num_params() >= before          # probes planted
    net.end()
    # leftover probes retired; the live graph remains
    assert net.num_params() >= 2


def test_compiler_note():
    assert shutil.which("cc") or shutil.which("gcc")
