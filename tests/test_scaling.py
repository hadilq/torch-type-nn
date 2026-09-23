"""Ports of the type-nn C test-suite (test_type_nn.c) for the scaling rules."""

import math

import torch
from helpers import D, assert_ulps

from torch_type_nn import AndOr, StructureScaler, TypeAdam, TypeNN, birth_depth, fit, mse_loss
from torch_type_nn.scaling import DONE, FIT, GROW, PRUNE, phase_at


def make(n, m, seed, lr, n_train, epochs, batch=1):
    net = TypeNN(n, m, seed=seed, dtype=D)
    opt = TypeAdam(net, lr=lr)
    sc = StructureScaler(net, opt, epochs=epochs, steps_per_epoch=math.ceil(n_train / batch))
    return net, opt, sc


def run_epoch(net, opt, sc, X, Y, batch=1):
    for s in range(0, X.shape[0], batch):
        x, t = X[s:s + batch], Y[s:s + batch]
        y = net(x)
        opt.zero_grad()
        mse_loss(y, t).backward()
        opt.step()
        sc.observe(x, y, t)


def data(n_in, N, seed, scale=1.5):
    g = torch.Generator().manual_seed(seed)
    X = scale * (2 * torch.rand(N, n_in, dtype=D, generator=g) - 1)
    y0 = (X[:, 0] * X[:, 1] > 0).to(D)
    return X, torch.stack([y0, 1 - y0], 1)


def test_birth_depth():
    assert [birth_depth(*s) for s in [(2, 1), (4, 3), (13, 3), (30, 1), (10, 1), (34, 1), (1, 1)]] \
        == [1, 3, 4, 3, 2, 4, 1]
    net = TypeNN(4, 3, dtype=D)
    assert net.depth == 3 and net.structure() == [[1, 1, 1]] * 3


def test_probes_are_exact_identities():
    a, _, sa = make(5, 2, 17, 0.0, 10, 10)
    b = TypeNN(5, 2, seed=17, dtype=D)
    sa.begin()                      # probes with noise amplitude lr = 0
    assert a.num_params() > b.num_params()
    X = 2 * torch.rand(20, 5, dtype=D) - 1
    assert_ulps(a(X), b(X))


def test_distance_from_identity():
    l = AndOr(3, 3, 1, dtype=D)
    with torch.no_grad():
        l.weight.zero_()
        l.bias.zero_()
        for k in range(3):
            l.weight[k, 0, k] = 1.0             # carrier
            l.add_or(k)                          # x 1
    assert all(StructureScaler.dev_or(l, k, 1) == 0.0 for k in range(3))
    assert StructureScaler.dev_layer(l) == 0.0
    with torch.no_grad():
        l.weight[1, 0, 2] = 0.5
    assert StructureScaler.dev_layer(l) > 0.0
    assert StructureScaler.dev_column(l, 0) > 0.0
    assert math.isinf(StructureScaler.dev_layer(AndOr(3, 2, dtype=D)))


def test_schedule_and_threshold():
    assert phase_at(0.0) == GROW and phase_at(0.33) == GROW
    assert phase_at(0.34) == FIT and phase_at(0.66) == FIT
    assert phase_at(0.67) == PRUNE and phase_at(1.0) == PRUNE
    _, _, sc = make(2, 1, 1, 0.01, 100, 1)
    assert abs(sc.threshold(1600) / sc.threshold(100) - 8.0) < 1e-12
    assert 0.01 * math.sqrt(1600) < sc.threshold(1600) < 0.01 * 1600


def test_evidence_formulas():
    _, _, sc = make(2, 1, 1, 0.0, 100, 1)
    sc.n_train = 100
    n = 100.0
    assert sc.bic_ratio(0.1, 0.1, 5) == 0.0
    assert sc.bic_ratio(0.1, 0.05, 5) == 0.0
    assert abs(sc.bic_ratio(0.1, 0.2, 5) - n * math.log(2) / (5 * math.log(n))) < 1e-12
    assert sc.bic_ratio(0.1, 0.2, 5) > 1.0 > sc.bic_ratio(0.1, 0.2, 100)
    assert abs(sc.criterion(0.1, 7) - (n * math.log(0.1) + 7 * math.log(n))) < 1e-12


def test_measured_mse_is_the_mse_of_the_epoch():
    X, Y = data(3, 30, 21, scale=1.0)
    net, opt, sc = make(3, 2, 4, 0.0, 30, 10)
    sc.begin()
    run_epoch(net, opt, sc, X, Y)
    with torch.no_grad():
        direct = float(((net(X) - Y) ** 2).mean())
    assert sc._ep.n == 30
    assert abs(sc.measure_mse() - direct) < 1e-12


def test_residual_gate():
    _, _, sc = make(2, 1, 1, 0.0, 4, 1)
    sc.n_train, sc.epoch_base = 4, 0.25
    sc.epoch_loss = 0.25 / 4 + 1e-9
    assert sc.residual_unexplained()
    sc.epoch_loss = 0.25 / 4 - 1e-9
    assert not sc.residual_unexplained()


def test_prune_never_makes_the_model_worse():
    X, Y = data(4, 60, 8)
    E = 60
    net, opt, sc = make(4, 2, 12, 0.005, 60, E)
    sc.begin()

    def crit():
        with torch.no_grad():
            m = float(((net(X) - Y) ** 2).mean())
        n = 60.0 * 2
        return n * math.log(m) + net.num_params() * math.log(n)

    checked = bad = 0
    for _ in range(E):
        run_epoch(net, opt, sc, X, Y)
        pre, best = crit(), sc.crit_best
        sc.epoch_end()
        if sc.phase == PRUNE:
            post = crit()
            ref = max(pre, best)
            checked += 1
            bad += post > ref + 1e-9 * abs(ref)
    assert checked > 0 and bad == 0


def test_depth_fold_beats_a_naive_insert():
    X, Y = data(3, 64, 3)
    a, oa, sa = make(3, 2, 9, 0.0, 64, 30)
    sa.begin()
    run_epoch(a, oa, sa, X, Y)              # lr 0: statistics and residual only
    with torch.no_grad():
        before = a(X)
    b = TypeNN(3, 2, seed=9, dtype=D)
    ident = AndOr(3, 3, 1, dtype=D)
    with torch.no_grad():
        ident.weight.zero_()
        ident.bias.zero_()
        for k in range(3):
            ident.weight[k, 0, k] = 1.0
    b.insert_layer(0, ident)
    d0 = a.depth
    sa.epoch_end()
    assert a.depth == d0 + 1
    with torch.no_grad():
        err_fold = float((a(X) - before).abs().sum())
        err_naive = float((b(X) - before).abs().sum())
    assert err_fold < err_naive


def test_xor_fits():
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=D)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=D)
    net = TypeNN(2, 1, seed=34972, dtype=D)
    fit(net, X, Y, epochs=2000, lr=0.008, shuffle=False)
    with torch.no_grad():
        y = net(X)
    assert float(((y - Y) ** 2).mean()) < 1e-3
    assert bool(((y >= 0.5) == (Y >= 0.5)).all())


def _check_invariants(net, n_in, n_out):
    L = net.layers
    assert L[0].in_features == n_in and L[-1].out_features == n_out
    assert all(L[i].out_features == L[i + 1].in_features for i in range(net.depth - 1))
    assert not any(l.probe for l in L)
    assert not any(bool(l.unit_probe.any()) or bool((l.or_probe & l.mask).any()) for l in L)
    assert all(bool((l.num_ors() >= 1).all()) and l.out_features >= 1 for l in L)
    for l in L:
        live = l.mask
        assert torch.isfinite(l.weight).all() and torch.isfinite(l.bias).all()
        assert bool((l.assembly[live] >= 1).all())
        assert bool(l.mask.any(0).all()), "compacted"
    assert net.num_params() == sum(int(l.mask.sum()) * (l.in_features + 2) for l in L)


def test_invariants_after_training():
    g = torch.Generator().manual_seed(77)
    X = 2 * torch.rand(80, 6, dtype=D, generator=g) - 1
    y0 = (X[:, 0] * X[:, 1] + 0.3 > 0.3).to(D)
    Y = torch.stack([y0, 1 - y0], 1)
    net = TypeNN(6, 2, seed=5, dtype=D)
    res = fit(net, X, Y, epochs=60, lr=0.005, generator=torch.Generator().manual_seed(1))
    _check_invariants(net, 6, 2)
    c = res.counters
    assert c.and_add + c.or_add + c.layer_add > 0, "something grew"


def test_batched_training_and_checkpoint_roundtrip():
    g = torch.Generator().manual_seed(3)
    X = torch.randn(256, 5, dtype=D, generator=g)
    Y = torch.stack([(X[:, 0] > 0).to(D), (X[:, 1] * X[:, 2] > 0).to(D)], 1)
    net = TypeNN(5, 2, seed=2, dtype=D)
    fit(net, X, Y, epochs=30, lr=0.01, batch_size=32, generator=g)
    _check_invariants(net, 5, 2)
    other = TypeNN(5, 2, seed=99, dtype=D)
    other.load_state_dict(net.state_dict())
    with torch.no_grad():
        assert torch.equal(other(X), net(X))
    assert other.structure() == net.structure()


def test_scaler_is_done_after_end():
    X, Y = data(3, 20, 1)
    net, opt, sc = make(3, 2, 1, 0.005, 20, 3)
    sc.begin()
    for _ in range(3):
        run_epoch(net, opt, sc, X, Y)
        sc.epoch_end()
    sc.end()
    assert sc.phase == DONE
    assert not any(l.track_stats for l in net.layers)


def test_pickle_and_deepcopy(tmp_path):
    import copy

    net = TypeNN(3, 2, seed=4, dtype=D)
    X = torch.randn(5, 3, dtype=D)
    c = copy.deepcopy(net)
    torch.save(net, tmp_path / "m.pt")
    back = torch.load(tmp_path / "m.pt", weights_only=False)
    with torch.no_grad():
        assert torch.equal(c(X), net(X)) and torch.equal(back(X), net(X))
    assert torch.equal(back.generator.get_state(), net.generator.get_state())


def test_prune_boundary_respects_the_exact_criterion():
    """Regression: an iris run (seed 0 of the board) just before its first
    prune boundary. The greedy pool once scored trial models with a parameter
    tally that counted an Or twice (alone, then with its coordinate), and
    collapsed this 198-parameter network to 21 parameters at 50x the MSE."""
    from pathlib import Path

    fx = torch.load(Path(__file__).parent / "data" / "iris_prune_boundary.pt",
                    weights_only=False)
    net = TypeNN(4, 3, dtype=D)
    net.load_state_dict(fx["state_dict"])
    for key, v in fx["stats"].items():
        i, name = key.split(".", 1)
        getattr(net.layers[int(i)], name).copy_(v)
    opt = TypeAdam(net, lr=fx["lr"])
    sc = StructureScaler(net, opt, epochs=1, steps_per_epoch=1)
    sc.step, sc.total, sc.crit_best = fx["step"], fx["total"], fx["crit_best"]
    sc._ep.x, sc._ep.t, sc._ep.n = [fx["X"]], [fx["T"]], fx["X"].shape[0]
    before_mse = sc.measure_mse()
    before = sc.criterion(before_mse, net.num_params())
    assert net.num_params() == 198
    sc._prune_step()
    after = sc.criterion(sc.measure_mse(), net.num_params())
    assert after <= before, (before, after)
    assert net.num_params() > 21
    _check_prunable_structure(net)


def _check_prunable_structure(net):
    L = net.layers
    assert all(L[i].out_features == L[i + 1].in_features for i in range(net.depth - 1))
    assert all(bool((l.num_ors() >= 1).all()) for l in L)
