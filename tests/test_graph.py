"""Tests for the graph (slime-mould) variant."""

import torch

from torch_type_nn import GraphTypeNN
from torch_type_nn.network import birth_depth


def test_birth_depth_and_small_world():
    n, m = 4, 3
    g = GraphTypeNN(n, m, seed=1)
    assert len(g.nodes) == birth_depth(n, m)
    assert g.nodes[0].in_features == n
    # every later node has at least one incoming (backbone)
    for node in g.nodes[1:]:
        assert len(node.incoming) >= 1
        assert node.in_features == len(node.incoming) * m
    # forward shape
    x = torch.randn(5, n)
    y = g(x)
    assert y.shape == (5, m)


def test_multi_source_concatenation():
    # force a known graph: 3 nodes, edges 0->1, 0->2, 1->2
    g = GraphTypeNN(2, 2, depth=3, k=1, p=0.0, seed=0)
    # with p=0 and backbone we should have at least the chain; check fan-in of last
    assert g.nodes[2].in_features >= 2
    x = torch.randn(4, 2)
    y = g(x)
    assert y.shape == (4, 2)
    # identity: zeroing a dummy block (if we had one) would not change output;
    # for now just check that the forward is deterministic
    y2 = g(x)
    assert torch.allclose(y, y2)


def test_structure_view():
    g = GraphTypeNN(8, 4, seed=42)
    s = g.structure()
    assert s["nodes"] == g.birth_depth
    assert s["edges"] >= g.birth_depth - 1
    assert s["params"] == g.num_params()
    assert len(s["fan_in"]) == s["nodes"]


def test_dummy_edge_preserves_function():
    g = GraphTypeNN(3, 2, depth=3, seed=1)
    x = torch.randn(6, 3)
    y_before = g(x).detach().clone()
    # add a dummy edge 0 -> 2 (skipping 1)
    g.add_dummy_edge(0, 2)
    y_after = g(x)
    assert torch.allclose(y_before, y_after, atol=1e-6)
    assert (0, 2) in g.edges
    assert 0 in g.nodes[2].incoming


def test_backprop_feedback_recorded():
    g = GraphTypeNN(4, 2, depth=2, seed=1)
    x = torch.randn(3, 4, requires_grad=True)
    y = g(x)
    loss = y.sum()
    loss.backward()
    # at least the last node should have seen a gradient
    assert g.nodes[-1].backprop_mag > 0
    # source of the last node should have an edge_backprop entry
    srcs = g.nodes[-1].incoming
    if srcs:
        assert g.nodes[-1].node_id in g.nodes[srcs[0]].edge_backprop


def test_graph_scaler_schedule():
    import torch
    from torch.optim import Adam
    from torch_type_nn import GraphTypeNN, GraphScaler
    m = GraphTypeNN(4, 2, depth=3, seed=1)
    opt = Adam(m.parameters(), lr=0.01)
    scaler = GraphScaler(m, opt, epochs=3, steps_per_epoch=2)
    scaler.begin()
    for _ in range(2):
        x = torch.randn(2, 4)
        y = m(x)
        t = torch.randn(2, 2)
        loss = ((y - t) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        scaler.observe(x, y, t)
    scaler.epoch_end()
    assert hasattr(scaler.counters, "edge_add")
