"""graph-type-nn: the torch module, the C reference and the packed device store.

The three implementations share one rule (``graph/core.py``, a port of
type-nn's ``type_nn_graph.c``) and C's xorshift32 stream, so they are held to
the C reference number for number where the arithmetic allows it:

* birth: the same weights, the same small-world shortcuts, the same forward;
* one training step from the same state (``snapshot`` -> ``from_snapshot``):
  the same weights and Adam moments after the step, to 1e-12;
* the edits the rule makes are exact (dummy edge, width dummy, edge-weight
  re-estimation) or near-exact (depth, folded);
* whole runs: the same structure as C on small problems.
"""

import math

import pytest
import torch

from torch_type_nn import (
    CudaGraphTypeNN,
    GraphScaler,
    GraphTypeNN,
    TypeAdam,
    fit_graph,
    get_backend,
    mse_loss,
)
from torch_type_nn.graph.core import X, birth_depth
from torch_type_nn.native import native_available

D = torch.float64
NATIVE, NATIVE_DETAIL = native_available()
needs_c = pytest.mark.skipif(not NATIVE, reason=f"C backend unavailable: {NATIVE_DETAIL}")
SEED = 34972 + 7919


def _native(*a, **k):
    from torch_type_nn import NativeGraphTypeNN
    return NativeGraphTypeNN(*a, **k)


def toy(n=48, n_in=5, n_out=2, seed=3):
    g = torch.Generator().manual_seed(seed)
    Xd = torch.rand(n, n_in, generator=g, dtype=D) * 2 - 1
    Yd = torch.stack([torch.sin(2 * Xd[:, k % n_in]) * Xd[:, (k + 1) % n_in]
                      for k in range(n_out)], 1)
    return Xd, Yd


def xor():
    Xd = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=D)
    return Xd, torch.tensor([[0], [1], [1], [0]], dtype=D)


# ------------------------------------------------------------------ birth

@pytest.mark.parametrize("n_in,n_out", [(2, 1), (4, 3), (13, 3), (34, 1)])
def test_birth_is_dense_small_world(n_in, n_out):
    g = GraphTypeNN(n_in, n_out, seed=SEED)
    D_ = birth_depth(n_in, n_out)
    assert g.depth == D_ == g.birth_depth
    assert all(n.n_out == n_out for n in g.core.nodes)        # as dense as the chain
    assert g.core.nodes[0].n_in == n_in and g.core.reads_x(0)
    if D_ >= 3:
        assert g.n_edges > D_                                 # shortcuts exist
    shortcuts = [e for e in g.edges if e.probe]
    assert all(e.c == 0.0 for e in shortcuts)                 # born as dummies
    ok, why = g.check()
    assert ok, why


@needs_c
@pytest.mark.parametrize("n_in,n_out", [(2, 1), (4, 3), (13, 3), (30, 1)])
def test_birth_equals_c(n_in, n_out):
    c = _native(n_in, n_out, seed=SEED)
    t = GraphTypeNN(n_in, n_out, seed=SEED)
    p = CudaGraphTypeNN(n_in, n_out, seed=SEED, device="cpu")
    assert t.structure() == p.structure()
    snap = c.snapshot()
    assert [len(n["units"]) for n in snap["nodes"]] == [n.n_out for n in t.core.nodes]
    assert [(e["src"], e["dst"]) for e in snap["edges"]] == [(e.src, e.dst) for e in t.edges]
    x = torch.rand(7, n_in, dtype=D) * 2 - 1
    yc = c(x)
    with torch.no_grad():
        assert torch.allclose(t.eval()(x), yc, atol=1e-12, rtol=0)
    assert torch.allclose(p(x), yc, atol=1e-12, rtol=0)


@needs_c
def test_begin_draws_the_same_dummies():
    """begin() adds the Or and And dummies from the same xorshift stream."""
    c = _native(13, 3, seed=SEED)
    c.begin(20, 30, 0.03)
    p = CudaGraphTypeNN(13, 3, seed=SEED, device="cpu")
    p.begin(20, 30, 0.03)
    t = GraphTypeNN(13, 3, seed=SEED)
    opt = TypeAdam(t, lr=0.003)
    GraphScaler(t, opt, epochs=30, steps_per_epoch=20).begin()
    x = torch.rand(5, 13, dtype=D)
    yc = c(x)
    assert torch.allclose(p(x), yc, atol=1e-12, rtol=0)
    with torch.no_grad():
        assert torch.allclose(t.eval()(x), yc, atol=1e-12, rtol=0)
    assert t.structure() == p.structure() == c.structure()


# --------------------------------------------------------------- one step

@needs_c
@pytest.mark.parametrize("epochs_before", [0, 3])
def test_one_step_equals_c(epochs_before):
    """From the same state, one sample: forward, backward, per-Or Adam with the
    lr / c column scale. Every weight and moment agrees with C to 1e-12."""
    Xd, Yd = toy()
    c = _native(5, 2, seed=SEED)
    c.begin(Xd.shape[0], 30, 0.05)
    for ep in range(epochs_before):
        c.epoch(Xd, Yd, 1000 + ep)
    snap = c.snapshot()
    t = GraphTypeNN.from_snapshot(snap)
    p = CudaGraphTypeNN.from_snapshot(snap, device="cpu")
    x, y = Xd[5], Yd[5]
    # C
    c.train(True)
    c.step(x, y)
    after = c.snapshot()
    # packed store (hand-written backward)
    p.train(True)
    p.step(x, y)
    # torch module (autograd + TypeAdam)
    opt = TypeAdam(t, lr=snap["scalars"]["lr"])
    t.train()
    t.core.training = True
    out = t(x.unsqueeze(0))
    loss = mse_loss(out, y.unsqueeze(0))
    opt.zero_grad()
    loss.backward()
    opt.step()
    for v, n in enumerate(after["nodes"]):
        for k, u in enumerate(n["units"]):
            for r, o in enumerate(u["ors"]):
                for store in (t.core.nodes[v].store, p.core.nodes[v].store):
                    w, b, a = store.get_or(k, r)
                    assert torch.allclose(w, o["w"], atol=1e-12, rtol=0), (v, k, r)
                    assert abs(b - o["b"]) < 1e-12 and abs(a - o["a"]) < 1e-12
    # the feedback the step accumulated
    for e_c, e_p in zip(after["edges"], p.core.edges, strict=True):
        assert abs(e_c["feedback"][3] - float(e_p.ep_in)) < 1e-12      # inference
        assert abs(e_c["feedback"][4] - float(e_p.ep_out)) < 1e-12     # back-prop
        assert abs(e_c["feedback"][5] - float(e_p.ep_grad)) < 1e-12    # junction


# ------------------------------------------------------------ exact edits

def _out(model, x):
    with torch.no_grad():
        return (model.eval()(x) if isinstance(model, GraphTypeNN) else model(x)).clone()


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_dummies_and_reweighting_keep_the_function(backend, device):
    if backend == "torch":
        m = GraphTypeNN(6, 3, seed=SEED, device=device)
    else:
        m = CudaGraphTypeNN(6, 3, seed=SEED, device=device)
    core = m.core
    x = torch.rand(9, 6, dtype=D, device=device) * 2 - 1
    y0 = _out(m, x)
    assert core.add_width_probe(0)
    for e in core.edges:                       # fake lifetime feedback: c != 1
        e.in_acc, e.out_acc, e.acc_n = 0.3 + e.dst, 0.7 + (e.src % 3), 1.0
    core.renormalize()
    assert any(0.0 < e.c < 1.0 for e in core.edges)
    added = any(core.add_dummy_edge(u, v) for v in range(1, m.depth) for u in range(v))
    y1 = _out(m, x)
    assert torch.allclose(y0, y1, atol=1e-12, rtol=0), float((y0 - y1).abs().max())
    full = all(core.edge_find(u, v) >= 0 for v in range(1, m.depth) for u in range(v))
    assert added or full
    ok, why = core.check()
    assert ok, why


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_depth_dummy_is_near_exact_and_removable(backend):
    """An identity node on an edge emits F(x) (the partition function), so
    the folded insert is near-exact, not exact; removing it again folds back."""
    Xd, Yd = toy(n=64, n_in=6, n_out=3)
    if backend == "torch":
        m = GraphTypeNN(6, 3, seed=SEED)
    else:
        m = CudaGraphTypeNN(6, 3, seed=SEED, device="cpu")
    core = m.core
    core.begin(64, 10, 0.005)
    for i in range(64):                     # fill the evidence cache
        core.observe(Xd[i:i + 1], _out(m, Xd[i:i + 1]), Yd[i:i + 1])
    y0 = _out(m, Xd)
    nn = m.depth
    assert core.insert_node(core.loudest_edge()[0] if core.loudest_edge()[0] >= 0 else 0)
    assert m.depth == nn + 1 and core.depth_probe() >= 0
    y1 = _out(m, Xd)
    rel = float((y1 - y0).abs().mean() / y0.abs().mean())
    assert rel < 0.1, rel
    assert core.remove_node(core.depth_probe())
    assert m.depth == nn
    y2 = _out(m, Xd)
    assert float((y2 - y0).abs().mean() / y0.abs().mean()) < 0.1
    ok, why = core.check()
    assert ok, why


def test_gradcheck_through_edges_and_gains():
    m = GraphTypeNN(4, 3, seed=SEED)
    core = m.core
    core.add_width_probe(0)
    for e in core.edges:
        e.in_acc, e.out_acc, e.acc_n = 0.5 + e.dst, 0.2 + e.src % 2, 1.0
    core.renormalize()
    with torch.no_grad():                          # move every dummy off zero
        for n in core.nodes:
            n.store.L.weight.add_(0.05 * torch.randn_like(n.store.L.weight)
                                  * n.store.L.mask.unsqueeze(-1))
    x = (torch.rand(3, 4, dtype=D) * 2 - 1).requires_grad_(True)
    m.eval()
    assert torch.autograd.gradcheck(lambda t: m(t), (x,), eps=1e-6, atol=1e-7)


def test_random_edits_keep_every_invariant():
    Xd, Yd = toy(n=32, n_in=5, n_out=2)
    m = CudaGraphTypeNN(5, 2, seed=SEED, device="cpu")
    core = m.core
    core.begin(32, 10, 0.005)
    for i in range(32):
        core.observe(Xd[i:i + 1], m(Xd[i:i + 1]), Yd[i:i + 1])
    g = torch.Generator().manual_seed(1)

    def ri(n):
        return int(torch.randint(max(n, 1), (1,), generator=g))
    for _ in range(300):
        V, E = m.depth, len(core.edges)
        op = ri(6)
        if op == 0:
            u, v = sorted((ri(V), ri(V)))
            core.add_dummy_edge(u, v)
        elif op == 1:
            core.add_width_probe(ri(V))
        elif op == 2 and E:
            core.insert_node(ri(E))
        elif op == 3:
            core.remove_node(ri(V))
        elif op == 4 and E:
            i = ri(E)
            if core.edges[i].src != X and core.in_degree(core.edges[i].dst) > 1:
                core.remove_edge(i)
        else:
            for e in core.edges:
                e.in_acc += float(torch.rand(1, generator=g))
                e.out_acc += float(torch.rand(1, generator=g))
                e.acc_n += 1.0
        core.renormalize()
        ok, why = core.check()
        assert ok, why
        assert torch.isfinite(m(Xd)).all()


# -------------------------------------------------------------- training

def test_scaler_needs_typeadam_and_the_bic_rule():
    m = GraphTypeNN(2, 1, seed=SEED)
    with pytest.raises(TypeError, match="TypeAdam"):
        GraphScaler(m, torch.optim.Adam(m.parameters()), epochs=1, steps_per_epoch=1)
    with pytest.raises(ValueError, match="BIC"):
        GraphScaler(m, TypeAdam(m, lr=0.1), epochs=1, steps_per_epoch=1, rule="threshold")


# The C reference itself ends at the constant (mse 0.25) on about 1 xor seed
# in 8 (5 of seeds 1..40 under fit's shuffle). Per-sample training is chaotic,
# so a torch build with different rounding (another BLAS, another version)
# can move any one seed into that basin. A port fits xor if one of three
# seeds does: a false failure has probability ~(1/8)^3.
XOR_SEEDS = (SEED, SEED + 1, SEED + 2)


def test_fit_graph_fits_xor():
    Xd, Yd = xor()
    tried = []
    for seed in XOR_SEEDS:
        m = GraphTypeNN(2, 1, seed=seed)
        scaler = fit_graph(m, Xd, Yd, epochs=2000, lr=0.008, shuffle=False)
        with torch.no_grad():
            mse = float(((m(Xd) - Yd) ** 2).mean())
        ok, why = m.check()
        assert ok, why
        tried.append(mse)
        if mse < 1e-3:
            assert scaler.counters.and_add > 0
            return
    pytest.fail(f"no seed fitted xor: {tried}")


def test_packed_store_fits_xor(device):
    Xd, Yd = xor()
    tried = []
    for seed in XOR_SEEDS:
        m = CudaGraphTypeNN(2, 1, seed=seed, device=device)
        m.fit(Xd, Yd, epochs=2000, lr=0.08)
        mse = float(((m(Xd.to(device)) - Yd.to(device)) ** 2).mean())
        ok, why = m.check()
        assert ok, why
        tried.append(mse)
        if mse < 1e-3:
            return
    pytest.fail(f"no seed fitted xor: {tried}")


@needs_c
@pytest.mark.parametrize("backend", ["cuda", "torch"])
def test_whole_run_follows_c(backend):
    """bench.c's protocol on a small problem: the run makes the same
    structural decisions as C (equal structure every epoch) and ends at the
    same loss, up to the rounding of a different summation order."""
    Xd, Yd = toy(n=40, n_in=4, n_out=2)
    E, lr = 24, 0.05
    c = _native(4, 2, seed=SEED)
    c.begin(40, E, lr)
    if backend == "cuda":
        p = CudaGraphTypeNN(4, 2, seed=SEED, device="cpu")
        p.begin(40, E, lr)
    else:
        p = GraphTypeNN(4, 2, seed=SEED)
        opt = TypeAdam(p, lr=lr * 0.1)
        sc = GraphScaler(p, opt, epochs=E, steps_per_epoch=40, n_train=40)
        sc.begin()
    from torch_type_nn.graph.core import xorshift32
    for ep in range(E):
        seed = (34972 ^ ((ep + 1) * 0x9E3779B9)) & 0xFFFFFFFF
        c.epoch(Xd, Yd, seed)
        if backend == "cuda":
            p.epoch(Xd, Yd, seed)
        else:
            order, s = list(range(40)), seed
            for i in range(40, 1, -1):
                s = xorshift32(s)
                j = s % i
                order[i - 1], order[j] = order[j], order[i - 1]
            p.train()
            for i in order:
                y = p(Xd[i:i + 1])
                loss = mse_loss(y, Yd[i:i + 1])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sc.observe(Xd[i:i + 1], y, Yd[i:i + 1])
            sc.epoch_end()
        assert p.structure() == c.structure(), f"epoch {ep}"
    if backend == "cuda":
        p.end()
    else:
        sc.end()
    c.end()
    assert p.structure() == c.structure()
    yp = _out(p, Xd)
    yc = c(Xd)
    assert torch.allclose(yp, yc, atol=1e-8, rtol=0), float((yp - yc).abs().max())
    assert p.counters() == c.counters()


def test_criterion_wraps_k_like_c_size_t():
    """C's criterion takes K as size_t; prune_pool can pass P - K - np < 0,
    which C wraps to ~1.8e19 (reject). The port must not accept instead."""
    core = GraphTypeNN(2, 1, seed=SEED).core
    core.n_train, core.n_out = 30, 1
    assert core.criterion(0.01, -7) == core.criterion(0.01, (1 << 64) - 7)
    assert core.criterion(0.01, -7) > 1e19
    assert core.criterion(0.01, 5) < core.criterion(0.01, 6)


@needs_c
def test_prune_epochs_from_c_state_equal_c():
    """Iris, seed 0: from C's own state at the start of each early prune
    epoch, the port takes C's decisions (structure and counters). The rule is
    the shared GraphCore; the torch store's arithmetic is pinned by
    test_one_step_equals_c. Epoch 166 is where prune_pool's size_t wrap
    first decides."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import bench
    try:
        Xtr, Ytr, _, _, _ = bench.split("iris")
    except FileNotFoundError:
        pytest.skip("iris.data not present")
    _, E, lr = bench.TASKS["iris"]
    c = _native(4, 3, seed=bench.SPLIT_SEED + 7919)
    c.begin(Xtr.shape[0], E, lr)

    def shuffle(ep):
        return bench.SPLIT_SEED ^ (((ep + 1) * 0x9E3779B9) & bench.MASK)

    for ep in range(166):
        c.epoch(Xtr, Ytr, shuffle(ep))
    for ep in range(166, 171):
        p = CudaGraphTypeNN.from_snapshot(c.snapshot(), device="cpu")
        c.epoch(Xtr, Ytr, shuffle(ep))
        p.epoch(Xtr, Ytr, shuffle(ep))
        assert p.structure() == c.structure(), f"epoch {ep}"
        assert p.counters() == c.counters(), f"epoch {ep}"


def test_backend_registry_graph():
    from torch_type_nn import NativeGraphTypeNN
    assert get_backend("torch", graph=True) is GraphTypeNN
    assert get_backend("c", graph=True) is NativeGraphTypeNN
    assert get_backend("cuda", graph=True) is CudaGraphTypeNN
    assert not math.isnan(0.0)


@needs_c
def test_native_board_cell_is_bench_c():
    """bench.py's C graph row is C bench.c's: iris, seed 0, digit for digit."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import bench
    try:
        Xtr, Ytr, Xte, Yte, _ = bench.split("iris")
    except FileNotFoundError:
        pytest.skip("iris.data not present")
    net, _ = bench.train("c-graph-type-nn", 4, 3, bench.SPLIT_SEED + 7919, Xtr, Ytr,
                         250, 0.05, 1, "cpu")
    assert net.num_params() == 89 and net.depth == 4 and net.n_edges == 5
    # type-nn bench.c (73aff3c + tie-breaks, -O2): iris graph-type-nn seed 0
    assert float(((net(Xtr) - Ytr) ** 2).mean()) == pytest.approx(4.5728243e-4, rel=1e-6)
    assert float(((net(Xte) - Yte) ** 2).mean()) == pytest.approx(0.016351886, rel=1e-7)
