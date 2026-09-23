"""Stock optimizers following structural edits, and the scaling protocol."""

import pytest
import torch
from helpers import D, rich_layer
from torch import nn

from torch_type_nn import StructureScaler, TypeNN, mse_loss
from torch_type_nn.adapters import TypeNNAdapter
from torch_type_nn.edits import Edit, follow_structure, grow_index, keep_index
from torch_type_nn.protocol import Scalable
from torch_type_nn.scaling import as_scalable


def _grow_rows(lin: nn.Linear) -> list[Edit]:
    """Add one output unit to a Linear; report the edits."""
    edits = []
    for name in ("weight", "bias"):
        old = getattr(lin, name)
        pad = torch.zeros((1, *old.shape[1:]), dtype=old.dtype)
        new = nn.Parameter(torch.cat([old.detach(), pad], 0))
        setattr(lin, name, new)
        edits.append(Edit(old, new, 0, grow_index(old.shape[0])))
    return edits


def _steps(model, opt, n=3):
    for _ in range(n):
        opt.zero_grad()
        model(torch.randn(4, 3, dtype=D)).pow(2).sum().backward()
        opt.step()


@pytest.mark.parametrize("make_opt", [
    lambda p: torch.optim.Adam(p, lr=0.01),
    lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
])
def test_state_is_remapped_on_growth(make_opt):
    torch.manual_seed(0)
    lin = nn.Linear(3, 2).double()
    opt = make_opt(lin.parameters())
    _steps(lin, opt)
    before = {k: v.clone() for k, v in opt.state[lin.weight].items() if torch.is_tensor(v)}
    edits = _grow_rows(lin)
    follow_structure(opt, lin, edits)
    st = opt.state[lin.weight]
    assert lin.weight in opt.state and all(e.old not in opt.state for e in edits)
    for k, v in before.items():
        if v.dim():
            assert st[k].shape == (3, 3)
            assert torch.equal(st[k][:2], v) and st[k][2].abs().sum() == 0
    _steps(lin, opt)                         # and it keeps training
    assert torch.isfinite(lin.weight).all()


def test_remap_on_selection_and_param_set_sync():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).double()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    _steps(model, opt)
    old = model[0].weight
    keep = torch.tensor([0, 2, 3])
    new = nn.Parameter(old.detach().index_select(0, keep))
    ref = opt.state[old]["exp_avg"].index_select(0, keep).clone()
    model[0].weight = new
    model.append(nn.Linear(2, 2).double())   # a new layer without an edit
    follow_structure(opt, model, [Edit(old, new, 0, keep_index(keep))])
    assert torch.equal(opt.state[new]["exp_avg"], ref)
    held = [p for g in opt.param_groups for p in g["params"]]
    assert {id(p) for p in held} == {id(p) for p in model.parameters()}
    del model[2]                              # a removed layer
    follow_structure(opt, model, [])
    held = [p for g in opt.param_groups for p in g["params"]]
    assert {id(p) for p in held} == {id(p) for p in model.parameters()}
    assert all(any(p is q for q in model.parameters()) for p in opt.state)


def test_andor_reports_its_edits():
    g = torch.Generator().manual_seed(2)
    l = rich_layer(3, 2, 1, g)
    opt = torch.optim.Adam(l.parameters(), lr=0.01)
    x = torch.randn(5, 3, dtype=D, generator=g)
    for _ in range(3):
        opt.zero_grad()
        l(x).sum().backward()
        opt.step()
    m_bias = opt.state[l.bias]["exp_avg"].clone()
    sink = []
    l.edit_sink = sink
    l.add_or(0)          # grows the slot dimension of weight, bias, assembly
    l.add_input()        # grows the input dimension of weight
    l.add_unit()         # grows the unit dimension
    l.edit_sink = None
    assert len(sink) == 3 + 1 + 3
    follow_structure(opt, l, sink)
    st = opt.state[l.bias]["exp_avg"]
    assert st.shape == l.bias.shape == (3, 2)
    assert torch.equal(st[:2, :1], m_bias) and st[2].abs().sum() == 0 and st[:, 1].abs().sum() == 0
    opt.zero_grad()
    l(torch.randn(5, 4, dtype=D)).sum().backward()
    opt.step()


def test_protocol():
    net = TypeNN(3, 2, dtype=D)
    assert isinstance(TypeNNAdapter(net), Scalable)
    assert isinstance(as_scalable(net), TypeNNAdapter)
    with pytest.raises(TypeError, match="Scalable"):
        as_scalable(nn.Linear(3, 2))


def test_scaling_with_a_stock_optimizer():
    """The whole grow/prune schedule driven by torch.optim.Adam: after every
    boundary the optimizer holds exactly the model's parameters, with state
    shaped like them."""
    g = torch.Generator().manual_seed(7)
    X = 2 * torch.rand(96, 4, dtype=D, generator=g) - 1
    y0 = (X[:, 0] * X[:, 1] > 0).to(D)
    Y = torch.stack([y0, 1 - y0], 1)
    net = TypeNN(4, 2, seed=3, dtype=D)
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    E, B = 60, 8
    sc = StructureScaler(net, opt, epochs=E, steps_per_epoch=96 // B)
    sc.begin()
    first = None
    for _ in range(E):
        for s in range(0, 96, B):
            x, t = X[s:s + B], Y[s:s + B]
            y = net(x)
            loss = mse_loss(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sc.observe(x, y, t)
        first = first if first is not None else sc._ep.loss_sum / sc._ep.n
        sc.epoch_end()
        held = [p for grp in opt.param_groups for p in grp["params"]]
        assert {id(p) for p in held} == {id(p) for p in net.parameters()}
        for p, st in opt.state.items():
            assert any(p is q for q in net.parameters())
            for v in st.values():
                assert not torch.is_tensor(v) or v.dim() == 0 or v.shape == p.shape
    sc.end()
    c = sc.counters
    assert c.and_add + c.or_add + c.layer_add > 0
    with torch.no_grad():
        final = float(((net(X) - Y) ** 2).mean())
    assert final < first
