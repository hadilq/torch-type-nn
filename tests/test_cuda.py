"""Ragged CUDA backend: runs on cpu here, and on cuda when a device is visible."""

import torch

from torch_type_nn import CudaTypeNN, available_backends, get_backend
from torch_type_nn.cuda import cuda_available


def test_backend_registry():
    names = available_backends()
    assert "cuda" in names
    assert get_backend("cuda") is CudaTypeNN
    assert names["cuda"].startswith("ready") or names["cuda"].startswith("unavailable")


def test_begin_forward_structure(device):
    net = CudaTypeNN(4, 3, rule="threshold", seed=7, device=device)
    assert net.depth == 0
    net.begin(8, 2, 0.05)
    y = net(torch.randn(4, dtype=torch.float64, device=device))
    assert y.shape == (3,) and torch.isfinite(y).all()
    assert y.device.type == torch.device(device).type
    assert net.structure() and net.num_params() > 0
    net.end()


def test_xor_fits_threshold(device):
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                     dtype=torch.float64, device=device)
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]],
                     dtype=torch.float64, device=device)
    net = CudaTypeNN(2, 1, rule="threshold", seed=34972, device=device)
    net.fit(X, Y, epochs=400, lr=0.08, shuffle=False)
    pred = net(X).reshape(-1) >= 0.5
    assert bool((pred == (Y.reshape(-1) >= 0.5)).all())


def test_ragged_not_padded(device):
    net = CudaTypeNN(3, 2, seed=1, device=device)
    net.begin(4, 2, 0.05)
    for layer in net.layers:
        assert layer.W.ndim == 2 and layer.W.shape[1] == layer.n_in
        assert layer.W.device.type == torch.device(device).type
        assert layer.W.shape[0] == sum(layer.num_ors())
    net.end()


def test_status_mentions_device():
    ok, detail = cuda_available()
    assert isinstance(ok, bool) and detail


def test_batched_forward_matches_row_loop(device):
    net = CudaTypeNN(3, 2, seed=3, device=device)
    net.begin(4, 1, 0.05)
    X = torch.randn(5, 3, dtype=torch.float64, device=device)
    batched = net(X)
    stacked = torch.stack([net(X[i]) for i in range(5)])
    assert torch.allclose(batched, stacked)
    net.end()


def test_bench_cuda_models_are_cudatypenn(device):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import bench
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                     dtype=torch.float64, device=device)
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]],
                     dtype=torch.float64, device=device)
    net, scaler = bench.train(
        "cuda-type-nn-overfit", 2, 1, 7, X, Y, 5, 0.08, 1, device)
    assert isinstance(net, CudaTypeNN) and scaler is None
    assert net.num_params() > 0
    assert net.layers[0].W.ndim == 2
