"""Every code path on every device: the ``device`` fixture (tests/conftest.py)
runs these on the CPU and, when CUDA is visible, on the GPU. The GPU check of
``flake.nix`` sets ``TNN_REQUIRE_CUDA=1`` so the CUDA half cannot be skipped.

CUDA results are compared with the CPU in float64. Reductions may run in a
different order on the GPU, so values are compared to a tolerance, never bit
for bit, and structure decisions (which depend on thresholds) are only checked
for their invariants.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
from helpers import D, rich_layer

from torch_type_nn import (
    ScalableMLP,
    StructureScaler,
    TypeAdam,
    TypeNN,
    fit,
    follow_structure,
    mse_loss,
)
from torch_type_nn.functional import and_or


def close(a, b, tol=1e-10):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float(((a - b).abs() / b.abs().clamp(min=1.0)).max()) < tol


def test_forward_and_backward_match_the_cpu(device):
    g = torch.Generator().manual_seed(0)
    ref = rich_layer(4, 3, 3, g)
    with torch.no_grad():
        ref.bias[1, 2] = 0.0
        ref.weight[1, 2] = 0.0                    # an exact zero factor (one-sided rule)
        ref.assembly[1, 2] = 1.0
    ref.add_or(2)                                 # a ragged unit: padding in use
    x = torch.randn(7, 4, dtype=D, generator=g)
    lay = rich_layer(4, 3, 3, torch.Generator().manual_seed(0)).to(device)
    lay.load_state_dict(ref.state_dict())
    lay.to(device)
    outs = []
    for l, xx in ((ref, x), (lay, x.to(device))):
        xx = xx.clone().requires_grad_(True)
        y = l(xx)
        (y * torch.arange(1, 4, dtype=D, device=y.device)).sum().backward()
        outs.append((y, xx.grad, l.weight.grad, l.bias.grad, l.assembly.grad))
    assert outs[1][0].device.type == device.type
    for a, b in zip(outs[1], outs[0], strict=True):
        assert torch.isfinite(a).all() and close(a, b)


def test_gradcheck(device):
    g = torch.Generator().manual_seed(1)
    l = rich_layer(3, 2, 2, g).to(device)
    x = torch.randn(5, 3, dtype=D, generator=g).to(device).requires_grad_(True)
    w = l.weight.detach().clone().requires_grad_(True)
    b = l.bias.detach().clone().requires_grad_(True)
    a = l.assembly.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x, w, b, a: and_or(x, w, b, a, l.mask),
                                    (x, w, b, a), eps=1e-6, atol=1e-6)


def test_type_adam_matches_the_cpu(device):
    g = torch.Generator().manual_seed(2)
    nets = [TypeNN(4, 2, seed=3, dtype=D), TypeNN(4, 2, seed=3, dtype=D).to(device)]
    X = torch.randn(16, 4, dtype=D, generator=g)
    Y = torch.rand(16, 2, dtype=D, generator=g)
    for net in nets:
        opt = TypeAdam(net, lr=0.01)
        dev = net.layers[0].weight.device
        for s in range(0, 16, 4):
            opt.zero_grad()
            mse_loss(net(X[s:s + 4].to(dev)), Y[s:s + 4].to(dev)).backward()
            opt.step()
    for a, b in zip(nets[1].layers, nets[0].layers, strict=True):
        assert close(a.weight, b.weight) and close(a.assembly, b.assembly)
        assert int(a.adam_step.sum()) == int(b.adam_step.sum())


def _data(device, n=96, n_in=5):
    g = torch.Generator().manual_seed(4)
    X = torch.randn(n, n_in, dtype=D, generator=g)
    Y = torch.stack([(X[:, 0] > 0).to(D), (X[:, 1] * X[:, 2] > 0).to(D)], 1)
    return X.to(device), Y.to(device)


def _invariants(net, n_in, n_out, device):
    L = net.layers
    assert L[0].in_features == n_in and L[-1].out_features == n_out
    assert all(L[i].out_features == L[i + 1].in_features for i in range(net.depth - 1))
    assert not any(l.probe or bool(l.unit_probe.any()) or bool((l.or_probe & l.mask).any())
                   for l in L)
    for l in L:
        for t in (*l.parameters(), *l.buffers()):
            assert t.device.type == device.type, "every tensor stays on the device"
        assert torch.isfinite(l.weight).all() and bool((l.assembly[l.mask] >= 1).all())


@pytest.mark.parametrize("rule", ["threshold", "bic"])
def test_type_nn_scales_on_the_device(device, rule):
    X, Y = _data(device)
    net = TypeNN(5, 2, seed=2, dtype=D).to(device)
    res = fit(net, X, Y, epochs=24, lr=0.01, batch_size=8, rule=rule,
              generator=torch.Generator().manual_seed(0))
    _invariants(net, 5, 2, device)
    c = res.counters
    assert c.or_add + c.and_add + c.layer_add > 0
    with torch.no_grad():
        assert torch.isfinite(net(X)).all()


@pytest.mark.parametrize("rule", ["threshold", "bic"])
def test_type_nn_with_a_stock_optimizer_on_the_device(device, rule):
    X, Y = _data(device)
    net = TypeNN(5, 2, seed=5, dtype=D).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    E, B = 24, 8
    sc = StructureScaler(net, opt, epochs=E, steps_per_epoch=math.ceil(len(X) / B), rule=rule)
    sc.begin()
    for _ in range(E):
        for s in range(0, len(X), B):
            x, t = X[s:s + B], Y[s:s + B]
            y = net(x)
            opt.zero_grad()
            mse_loss(y, t).backward()
            opt.step()
            sc.observe(x, y, t)
        sc.epoch_end()
        for st in opt.state.values():
            for v in st.values():
                assert not torch.is_tensor(v) or v.dim() == 0 or v.device.type == device.type
    sc.end()
    _invariants(net, 5, 2, device)


@pytest.mark.parametrize("rule", ["threshold", "bic"])
def test_scalable_mlp_on_the_device(device, rule):
    X, Y = _data(device)
    net = ScalableMLP(5, 2, seed=1, dtype=D).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    E, B = 30, 8
    sc = StructureScaler(net, opt, epochs=E, steps_per_epoch=math.ceil(len(X) / B), rule=rule)
    sc.begin()
    for _ in range(E):
        for s in range(0, len(X), B):
            x, t = X[s:s + B], Y[s:s + B]
            y = net(x)
            opt.zero_grad()
            mse_loss(y, t).backward()
            opt.step()
            sc.observe(x, y, t)
        sc.epoch_end()
    sc.end()
    for lin in net.linears:
        assert lin.weight.device.type == device.type and not lin.probe
    ws = [l.weight.shape for l in net.linears]
    assert all(ws[i][0] == ws[i + 1][1] for i in range(len(ws) - 1))


def test_follow_structure_keeps_state_on_the_device(device):
    net = TypeNN(3, 2, seed=1, dtype=D).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    mse_loss(net(torch.randn(4, 3, dtype=D, device=device)),
             torch.zeros(4, 2, dtype=D, device=device)).backward()
    opt.step()
    sink = []
    l = net.layers[0]
    l.edit_sink = sink
    l.add_or(0)
    l.add_input()
    l.edit_sink = None
    follow_structure(opt, net, sink)
    for p in net.parameters():
        for v in opt.state.get(p, {}).values():
            if torch.is_tensor(v) and v.dim():
                assert v.shape == p.shape and v.device.type == device.type


def test_checkpoint_moves_between_devices(device):
    X, Y = _data("cpu")
    net = TypeNN(5, 2, seed=2, dtype=D)
    fit(net, X, Y, epochs=12, lr=0.01, batch_size=16)
    other = TypeNN(5, 2, seed=9, dtype=D).to(device)
    other.load_state_dict(net.state_dict())
    other.to(device)
    with torch.no_grad():
        assert close(other(X.to(device)), net(X))


@pytest.mark.parametrize("kind", ["type-nn", "type-nn-bic", "ref-type-nn",
                                  "ref-type-nn-overfit", "mlp", "mlp-scaled"])
def test_every_benchmark_model_trains_on_the_device(device, kind):
    """The board's own training loop (benchmarks/bench.py), shortened."""
    sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
    import bench

    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=D, device=device)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=D, device=device)
    model, _ = bench.train(kind, 2, 1, 7, X, Y, 30, 0.08, 1, device)
    mse, acc = bench.metrics(model, X, Y, True)
    assert math.isfinite(mse) and acc is not None
