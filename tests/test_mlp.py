"""The MLP adapter against the protocol contract (docs/ADAPTERS.md)."""

import math

import torch
from helpers import D, assert_ulps

from torch_type_nn import MLPAdapter, ScalableMLP, StructureScaler, mse_loss
from torch_type_nn.protocol import DEGREE, DEPTH, WIDTH, EditContext, Item, Scalable
from torch_type_nn.scaling import PRUNE, as_scalable


def data(N=96, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = 2 * torch.rand(N, n, dtype=D, generator=g) - 1
    y0 = (X[:, 0] * X[:, 1] > 0).to(D)
    return X, torch.stack([y0, 1 - y0], 1)


def ctx(noise=0.0, step=0):
    return EditContext(step=step, noise=noise, generator=torch.Generator().manual_seed(9))


def test_birth_and_protocol():
    net = ScalableMLP(4, 3, dtype=D)
    assert net.structure() == [3] and net.num_params() == 3 * 5 + 3 * 4
    assert isinstance(as_scalable(net), MLPAdapter) and isinstance(MLPAdapter(net), Scalable)
    a = MLPAdapter(net)
    assert a.probe_sites(DEGREE) == [] and a.probe_at(DEGREE, None) is None


def test_probes_are_identities():
    X, _ = data()
    net = ScalableMLP(4, 2, [3, 3], dtype=D, seed=4)
    a = MLPAdapter(net)
    with torch.no_grad():
        before = net(X)
    a.add_probe(WIDTH, 0, ctx())
    a.add_probe(WIDTH, 1, ctx())
    a.add_probe(DEPTH, 2, ctx())            # gap after the second ReLU
    a.add_probe(DEPTH, 1, ctx())
    assert net.depth == 5
    a.add_probe(WIDTH, 0, ctx())            # through the depth probe at gap 1
    assert net.structure() == [5, 5, 4, 4]
    assert a.probe_sites(WIDTH) == [0, 2]     # 1 and 3 are depth probes
    with torch.no_grad():
        assert_ulps(net(X), before)


def test_width_probe_through_depth_probe_removes_exactly():
    X, _ = data()
    net = ScalableMLP(4, 2, dtype=D, seed=6)
    a = MLPAdapter(net)
    a.add_probe(DEPTH, 1, ctx())
    a.add_probe(WIDTH, 0, ctx())
    assert net.structure() == [3, 3]
    item = a.probe_at(WIDTH, 0)
    with torch.no_grad():
        a._consumer(0).weight[:, item.address[1]] = 0.3   # it learned something
        ref = net(X)
    undo = a.ablate(item)
    with torch.no_grad():
        ablated = net(X)
    undo()
    with torch.no_grad():
        assert torch.equal(net(X), ref)
    a.ablate(item)
    a._drop_coordinate(*item.address)
    assert net.structure() == [2, 2] and net.linears[1].weight.shape == (2, 2)
    with torch.no_grad():
        assert_ulps(net(X), ablated)


def test_ablation_undo_is_exact_and_commit_preserves_function():
    X, _ = data()
    net = ScalableMLP(4, 2, [5, 5], dtype=D, seed=2)
    a = MLPAdapter(net)
    with torch.no_grad():
        ref = net(X)
    for item in [*a.prune_candidates(), Item(DEPTH, (1,))]:
        undo = a.ablate(item)
        undo()
        with torch.no_grad():
            assert torch.equal(net(X), ref)
    assert a.ablate(Item(DEPTH, (0,))) is None, "no exact identity in front of layer 0"
    removed = [Item(WIDTH, (0, 1)), Item(WIDTH, (0, 3)), Item(WIDTH, (1, 0))]
    for it in removed:
        a.ablate(it)
    with torch.no_grad():
        ablated = net(X)
    K = a.params_without(removed)
    a.commit_removals(removed)
    assert net.structure() == [3, 4] and net.num_params() == K
    with torch.no_grad():
        assert_ulps(net(X), ablated)


def test_keep_one_unit_per_junction():
    net = ScalableMLP(3, 1, [2], dtype=D)
    a = MLPAdapter(net)
    first = Item(WIDTH, (0, 0))
    assert a.can_remove(first, [])
    assert not a.can_remove(Item(WIDTH, (0, 1)), [first])


def train(net, opt, X, Y, E, B=8):
    sc = StructureScaler(net, opt, epochs=E, steps_per_epoch=math.ceil(len(X) / B))
    sc.begin()
    checks = []
    for _ in range(E):
        for s in range(0, len(X), B):
            x, t = X[s:s + B], Y[s:s + B]
            y = net(x)
            loss = mse_loss(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sc.observe(x, y, t)
        u = sc.step / sc.total
        if u >= 2 / 3:
            sc._close_epoch_residual()
            before = sc.criterion(sc.measure_mse(), net.num_params())
            ref = before if math.isinf(sc.crit_best) else max(sc.crit_best, before)
            with sc._editing():          # the edit scope syncs the optimizer
                sc._prune_step()
            after = sc.criterion(sc.measure_mse(), net.num_params())
            checks.append(after <= ref + 1e-9 * abs(ref))
            sc.phase = PRUNE
            sc._reset_epoch()
        else:
            sc.epoch_end()
        held = [p for g in opt.param_groups for p in g["params"]]
        assert {id(p) for p in held} == {id(p) for p in net.parameters()}
    sc.end()
    return sc, checks


def test_grows_prunes_and_never_worsens_the_criterion():
    X, Y = data(128)
    net = ScalableMLP(4, 2, dtype=D, seed=1)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    sc, checks = train(net, opt, X, Y, E=60)
    c = sc.counters
    assert c.or_add > 0, "width grew"
    assert checks and all(checks)
    assert not any(bool(l.unit_probe.any()) or l.probe for l in net.linears)
    ws = [l.weight.shape for l in net.linears]
    assert all(ws[i][0] == ws[i + 1][1] for i in range(len(ws) - 1))
    with torch.no_grad():
        acc = float((net(X).argmax(1) == Y.argmax(1)).double().mean())
    assert acc > 0.8


def test_checkpoint_roundtrip():
    X, Y = data(64)
    net = ScalableMLP(4, 2, dtype=D, seed=3)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    train(net, opt, X, Y, E=15)
    other = ScalableMLP(4, 2, dtype=D, seed=8)
    other.load_state_dict(net.state_dict())
    with torch.no_grad():
        assert torch.equal(other(X), net(X))
