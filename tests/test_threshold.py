"""The threshold rule (the default): every decision comes from back-prop.

A probe is promoted when its displacement exceeds theta(T) = lr T^(3/4); in
the prune phase anything inside the band theta_band = lr S^(3/4) is dropped:
an And keeps its most-moved Or (two identity Ors -> one), a junction keeps its
most-moved coordinate, and one (near-)identity layer goes per boundary. These
tests pin that behaviour, the fact that no training pair is re-read, and two
fixes in the scaler (see docs/DIFFERENCES.md).
"""

import math

import pytest
import torch
from helpers import D, assert_ulps

from torch_type_nn import AndOr, StructureScaler, TypeAdam, TypeNN, fit, mse_loss
from torch_type_nn.adapters import TypeNNAdapter
from torch_type_nn.protocol import DEGREE, DEPTH, WIDTH, Item
from torch_type_nn.scaling import PRUNE


def scaler(net, lr=0.01, steps=100, epochs=10, rule="threshold", **kw):
    opt = TypeAdam(net, lr=lr)
    return StructureScaler(TypeNNAdapter(net, **kw), opt, epochs=epochs,
                           steps_per_epoch=steps, rule=rule)


def test_rule_is_validated_and_band_formula():
    net = TypeNN(3, 2, dtype=D)
    with pytest.raises(ValueError, match="rule"):
        scaler(net, rule="nope")
    sc = scaler(net, lr=0.01, steps=81)
    assert sc.rule == "threshold"
    assert abs(sc.band() - 0.01 * 81 ** 0.75) < 1e-15
    assert abs(sc.band() - sc.threshold(81)) < 1e-15    # theta_band = theta(S)


def test_threshold_rule_never_reads_training_pairs(monkeypatch):
    """No pair is stored and no forward pass is re-run: decisions use weights only."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(40, 4, dtype=D, generator=g)
    Y = torch.stack([(X[:, 0] > 0).to(D), (X[:, 1] * X[:, 2] > 0).to(D)], 1)
    net = TypeNN(4, 2, seed=1, dtype=D)

    def forbidden(self):
        raise AssertionError("the threshold rule must not re-evaluate training pairs")

    monkeypatch.setattr(StructureScaler, "measure_mse", forbidden)
    res = fit(net, X, Y, epochs=30, lr=0.01, batch_size=4, generator=g)
    c = res.counters
    assert c.or_add + c.and_add + c.layer_add > 0


def _identity_or(layer, k):
    return layer.add_or(k)                     # w = 0, b = 1, a = 1: displacement 0


def test_two_identity_ors_in_an_and_leave_one():
    net = TypeNN(3, 2, depth=1, seed=2, dtype=D)
    l = net.layers[0]
    with torch.no_grad():                       # unit 0: only identity-like Ors
        l.weight[0, 0] = 0.0
        l.bias[0, 0] = 1.0
    _identity_or(l, 0)
    _identity_or(l, 1)                          # unit 1: one real Or + one identity
    sc = scaler(net, lr=0.01, steps=100)
    assert net.structure() == [[2, 2]]
    with sc._editing():
        sc._prune_band()
    assert net.structure() == [[1, 1]], "an And keeps exactly one of its identity Ors"
    assert sc.counters.and_drop == 2


def test_a_moved_or_survives_the_band():
    net = TypeNN(3, 1, depth=1, seed=2, dtype=D)
    l = net.layers[0]
    r = _identity_or(l, 0)
    with torch.no_grad():
        l.weight[0, r, 0] = 5.0                 # far outside the band
    sc = scaler(net, lr=0.01, steps=100)
    with sc._editing():
        sc._prune_band()
    assert net.structure() == [[2]]


def test_identity_coordinate_is_dropped_but_one_is_kept():
    net = TypeNN(4, 1, depth=2, seed=3, dtype=D)   # 4 -> 1 -> 1
    a = TypeNNAdapter(net)
    l0, l1 = net.layers
    for _ in range(2):                              # two more coordinates on the junction
        k = l0.add_unit()
        l0.add_or(k, torch.randn(4, dtype=D), 0.3)
        l1.add_input()                              # read with weight exactly 0
    with torch.no_grad():
        l1.weight[0, 0, 1] = 2.0                    # coordinate 1 is used
    sc = scaler(net, lr=0.01, steps=100)
    with sc._editing():
        sc._prune_band()
    assert [l.out_features for l in net.layers] == [1, 1]
    assert a.displacement(Item(WIDTH, (0, 0))) > sc.band(), "the used coordinate is kept"
    # a junction whose every coordinate is in the band keeps one
    with torch.no_grad():
        l1 = net.layers[1]
        l1.weight.zero_()
    with sc._editing():
        sc._prune_band()
    assert [l.out_features for l in net.layers] == [1, 1]


def test_identity_layer_is_dropped_and_a_moved_one_is_not():
    net = TypeNN(3, 2, depth=1, seed=4, dtype=D)
    ident = AndOr(3, 3, 1, dtype=D)
    with torch.no_grad():
        ident.weight.zero_()
        ident.bias.zero_()
        for k in range(3):
            ident.weight[k, 0, k] = 1.0             # carrier: an identity layer
    net.insert_layer(0, ident)
    sc = scaler(net, lr=0.01, steps=100)
    with sc._editing():
        sc._prune_band()
    assert net.depth == 1 and sc.counters.layer_drop == 1

    net.insert_layer(0, ident)
    with torch.no_grad():
        ident.weight[0, 0, 1] = 3.0                 # moved far from the identity
    sc = scaler(net, lr=0.01, steps=100)
    with sc._editing():
        sc._prune_band()
    assert net.depth == 2 and sc.counters.layer_drop == 0


def test_unmoved_depth_probe_is_retired_and_moved_one_promoted():
    from torch_type_nn.protocol import EditContext

    for push, depth, added in [(0.0, 1, 0), (3.0, 2, 1)]:
        net = TypeNN(3, 2, depth=1, seed=5, dtype=D)
        sc = scaler(net, lr=0.01, steps=100)
        sc.target.add_probe(DEPTH, 0, EditContext(0, 0.0, torch.Generator()))
        with torch.no_grad():
            net.layers[0].weight[0, 0, 1] += push
        with sc._editing():
            sc._prune_band()
        assert net.depth == depth and sc.counters.layer_add == added
        assert not any(l.probe for l in net.layers)


def test_every_coordinate_dropped_on_a_junction_keeps_its_mean():
    """Regression: dropping a coordinate resets the consumer's statistics, so
    before the fix only the first dropped coordinate of a junction had its mean
    folded into the consumer's bias. Two constant coordinates, both dropped at
    one boundary, must leave the function exactly unchanged."""
    net = TypeNN(3, 1, depth=2, seed=6, dtype=D)     # 3 -> 1 -> 1
    l0, l1 = net.layers
    for c in (0.7, -1.3):                             # constant units: w = 0, b = c
        k = l0.add_unit()
        l0.add_or(k, torch.zeros(3, dtype=D), c)
        l1.add_input()
    with torch.no_grad():
        l1.weight[0, 0, 1] = 1e-3                     # inside the band, not zero
        l1.weight[0, 0, 2] = -2e-3
    a = TypeNNAdapter(net)
    X = torch.randn(64, 3, dtype=D)
    a.set_tracking(True)
    net.train()
    net(X).sum().backward()                           # statistics of a training pass
    with torch.no_grad():
        before = net(X)
    a.commit_removals([Item(WIDTH, (0, 1)), Item(WIDTH, (0, 2))], axes=(WIDTH,))
    assert [l.out_features for l in net.layers] == [1, 1]
    with torch.no_grad():
        assert_ulps(net(X), before, ulps=64)


def test_end_before_the_prune_phase_measures_on_real_pairs():
    """Regression (BIC rule): ``end`` called before the prune phase used to judge
    the model on no pairs (a stale loss), so only the parameter count mattered
    and everything prunable was pruned."""
    g = torch.Generator().manual_seed(11)
    X = 2 * torch.rand(60, 4, dtype=D, generator=g) - 1
    y0 = (X[:, 0] * X[:, 1] > 0).to(D)
    Y = torch.stack([y0, 1 - y0], 1)
    net = TypeNN(4, 2, seed=12, dtype=D)
    opt = TypeAdam(net, lr=0.01)
    E = 30
    sc = StructureScaler(net, opt, epochs=E, steps_per_epoch=60, rule="bic")
    sc.begin()
    for _ in range(5):                                # stop inside the grow phase
        for s in range(60):
            x, t = X[s:s + 1], Y[s:s + 1]
            y = net(x)
            opt.zero_grad()
            mse_loss(y, t).backward()
            opt.step()
            sc.observe(x, y, t)
        sc.epoch_end()
    assert sc.phase != PRUNE and sc._pairs() is not None
    with torch.no_grad():
        direct = float(((net(X) - Y) ** 2).mean())
    assert abs(sc.measure_mse() - direct) < 1e-12, "end() sees the last epoch's pairs"
    pre = sc.criterion(direct, net.num_params())
    sc.end()
    with torch.no_grad():
        after = float(((net(X) - Y) ** 2).mean())
    assert sc.criterion(after, net.num_params()) <= pre + 1e-9 * abs(pre)


@pytest.mark.parametrize("rule", ["threshold", "bic"])
def test_both_rules_train_xor_and_leave_no_probe(rule):
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=D)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=D)
    net = TypeNN(2, 1, seed=34972, dtype=D)
    fit(net, X, Y, epochs=2000, lr=0.008, shuffle=False, rule=rule)
    with torch.no_grad():
        y = net(X)
    assert bool(((y >= 0.5) == (Y >= 0.5)).all())
    assert not any(l.probe or bool(l.unit_probe.any()) or bool((l.or_probe & l.mask).any())
                   for l in net.layers)


def test_width_grows_at_birth_depth_two_by_default():
    """The brief: width can grow on every junction. Born at depth 2 the C rule
    has no width site once a depth probe exists; the default has one."""
    g = torch.Generator().manual_seed(0)
    X = 2 * torch.rand(200, 10, dtype=D, generator=g) - 1
    Y = (torch.sin(3 * X[:, 0] * X[:, 1]) + X[:, 2] ** 2).unsqueeze(1) / 2
    counts = {}
    for through in (True, False):
        net = TypeNN(10, 1, seed=3, dtype=D)
        sc = scaler(net, lr=0.003, steps=math.ceil(200 / 16), epochs=30,
                    width_through_depth_probe=through)
        sc.begin()
        for _ in range(30):
            for s in range(0, 200, 16):
                x, t = X[s:s + 16], Y[s:s + 16]
                y = net(x)
                sc.optimizer.zero_grad()
                mse_loss(y, t).backward()
                sc.optimizer.step()
                sc.observe(x, y, t)
            sc.epoch_end()
        sc.end()
        counts[through] = sc.counters.or_add
    assert counts[True] > 0


def test_degree_items_are_judged_after_width_drops():
    """Width commits before degree is measured (as in C): an Or's distance
    from the identity is read on the structure the width pass left."""
    net = TypeNN(3, 1, depth=2, seed=7, dtype=D)
    l0, l1 = net.layers
    k = l0.add_unit()
    l0.add_or(k, torch.randn(3, dtype=D), 0.1)
    l1.add_input()                                    # new coordinate read with 0
    r = _identity_or(l1, 0)
    sc = scaler(net, lr=0.01, steps=100)
    seen = []
    orig = sc.target.displacement

    def spy(item):
        if item.axis == DEGREE:
            seen.append(net.layers[1].in_features)
        return orig(item)

    sc.target.displacement = spy
    with sc._editing():
        sc._prune_band()
    assert seen and all(n == 1 for n in seen), "degree judged after the coordinate went"
    assert r >= 0 and net.structure()[1] == [1]
