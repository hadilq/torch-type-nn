import shutil

import pytest
import torch

from torch_type_nn import NativeTypeNN, available_backends, get_backend
from torch_type_nn.native import native_available

ok, detail = native_available()
pytestmark = pytest.mark.skipif(not ok, reason=f"native backend unavailable: {detail}")


def test_backend_registry():
    names = available_backends()
    assert set(names) >= {"torch", "c", "cuda"}
    assert get_backend("c") is NativeTypeNN
    with pytest.raises(NotImplementedError):
        get_backend("cuda")


def test_begin_forward_structure():
    net = NativeTypeNN(4, 3, rule="threshold", seed=7)
    assert net.depth == 0
    with pytest.raises(RuntimeError, match="begin"):
        net(torch.zeros(4))
    net.begin(8, 2, 0.05)
    y = net(torch.randn(4, dtype=torch.float64))
    assert y.shape == (3,) and torch.isfinite(y).all()
    assert net.structure() and net.num_params() > 0
    net.end()


@pytest.mark.parametrize("rule", ["threshold", "bic"])
def test_xor_fits(rule):
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    net = NativeTypeNN(2, 1, rule=rule, seed=34972)
    net.fit(X, Y, epochs=400, lr=0.08, shuffle=False)
    pred = net(X).reshape(-1) >= 0.5
    assert bool((pred == (Y.reshape(-1) >= 0.5)).all())


def test_compiler_present():
    assert shutil.which("cc") or shutil.which("gcc")


def test_bench_c_models_are_nativetypenn():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import bench
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    net, scaler = bench.train("c-type-nn", 2, 1, 7, X, Y, 5, 0.08, 1, "cpu")
    assert isinstance(net, NativeTypeNN) and scaler is None
    assert net.num_params() > 0


def test_iris_one_seed_matches_c_bench():
    """NativeTypeNN + bench.split match type-nn bench.c on iris seed 0."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import bench
    try:
        Xtr, Ytr, Xte, Yte, _ = bench.split("iris")
    except FileNotFoundError:
        pytest.skip("iris.data not present")
    seed = bench.SPLIT_SEED + 7919
    net, _ = bench.train("c-type-nn", 4, 3, seed, Xtr, Ytr, 250, 0.05, 1, "cpu")
    train = float(((net(Xtr) - Ytr) ** 2).mean())
    hold = float(((net(Xte) - Yte) ** 2).mean())
    assert net.num_params() == 38 and net.depth == 3
    # C bench.c (7333bf7, -O2) on this seed. Last digits move with libc/cc;
    # the old Python-dy path landed at 2.74e-4 / 84 params — far outside this.
    assert train == pytest.approx(0.003759489919852301, rel=1e-12, abs=1e-12)
    assert hold == pytest.approx(0.028597245290, rel=1e-12, abs=1e-12)

