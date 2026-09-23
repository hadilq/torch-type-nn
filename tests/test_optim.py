import math

import torch
from helpers import D, rich_layer

from torch_type_nn import AndOr, TypeAdam

B1, B2, EPS = 0.9, 0.999, 1e-8


def scalar_adam(p, g, m, v, t, lr):
    """The reference: common.h tnn_adam on one scalar."""
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    c1, c2 = 1 / (1 - B1 ** t), 1 / (1 - B2 ** t)
    return p - lr * (m * c1) / (math.sqrt(v * c2) + EPS), m, v


def test_matches_scalar_adam_and_per_or_step():
    g = torch.Generator().manual_seed(0)
    l = rich_layer(3, 2, 1, g, a_max=1.0)
    opt = TypeAdam(l, lr=0.01)
    x = torch.randn(1, 3, dtype=D, generator=g)
    ref = {}
    for step in range(6):
        if step == 3:                       # an Or born late
            l.add_or(1, torch.full((3,), 0.2, dtype=D), 0.7, 1.3)
        opt.zero_grad()
        (l(x) ** 2).sum().backward()
        grads = {k: (l.bias.grad[k].item(), l.assembly.grad[k].item())
                 for k in [(0, 0), (1, 0), (1, 1)] if k[1] < l.degree and bool(l.mask[k])}
        before = {k: (l.bias[k].item(), l.assembly[k].item()) for k in grads}
        opt.step()
        for k, (gb, ga) in grads.items():
            st = ref.setdefault(k, {"t": 0, "mb": 0.0, "vb": 0.0, "ma": 0.0, "va": 0.0})
            st["t"] += 1
            b, st["mb"], st["vb"] = scalar_adam(before[k][0], gb, st["mb"], st["vb"], st["t"], 0.01)
            a, st["ma"], st["va"] = scalar_adam(before[k][1], ga, st["ma"], st["va"], st["t"], 0.01)
            assert abs(l.bias[k].item() - b) < 1e-14
            assert abs(l.assembly[k].item() - max(a, 1.0)) < 1e-14
            assert int(l.adam_step[k]) == st["t"]
    assert int(l.adam_step[1, 1]) == 3, "late Or counts from its own birth"


def test_assembly_stays_at_least_one():
    l = AndOr(2, 1, 1, dtype=D)
    opt = TypeAdam(l, lr=0.5)
    for _ in range(20):
        opt.zero_grad()
        l.assembly.sum().mul(1.0).backward()   # always pushes a down
        opt.step()
    assert float(l.assembly.detach().min()) >= 1.0


def test_structure_edits_between_steps():
    g = torch.Generator().manual_seed(1)
    l = rich_layer(3, 2, 1, g)
    opt = TypeAdam(l, lr=0.01)
    x = torch.randn(4, 3, dtype=D, generator=g)
    for step in range(5):
        opt.zero_grad()
        l(x[:, : l.in_features]).sum().backward()
        opt.step()
        if step == 1:
            l.add_input()
            x = torch.cat([x, torch.randn(4, 1, dtype=D, generator=g)], 1)
        if step == 2:
            l.add_unit()
            l.add_or(2)
    assert l.adam_m_weight.shape == l.weight.shape
    assert torch.isfinite(l.weight).all()
    assert int(l.adam_step[0, 0]) == 5 and int(l.adam_step[2, 0]) == 2


def test_other_parameters_get_plain_adam():
    torch.manual_seed(0)
    head_a = torch.nn.Linear(3, 2).double()
    head_b = torch.nn.Linear(3, 2).double()
    head_b.load_state_dict(head_a.state_dict())
    model = torch.nn.Sequential(AndOr(4, 3, dtype=D), head_a)
    ta = TypeAdam(model, lr=0.01)
    tb = torch.optim.Adam(head_b.parameters(), lr=0.01, foreach=False)
    x = torch.randn(5, 4, dtype=D)
    for _ in range(4):
        ta.zero_grad()
        tb.zero_grad()
        h = model[0](x).detach()
        model[1](h).pow(2).sum().backward()
        head_b(h).pow(2).sum().backward()
        ta.step()
        tb.step()
    for pa, pb in zip(head_a.parameters(), head_b.parameters(), strict=True):
        assert torch.allclose(pa, pb, rtol=1e-12, atol=1e-15)


def test_stock_optimizer_keeps_padding_identity():
    l = AndOr(3, 2, 1, dtype=D)
    l.add_or(0, torch.full((3,), 0.3, dtype=D))  # unit 1 now has a free slot
    opt = torch.optim.Adam(l.parameters(), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        l(torch.randn(4, 3, dtype=D)).sum().backward()
        opt.step()
    assert l.weight[1, 1].abs().sum() == 0 and l.bias[1, 1] == 1 and l.assembly[1, 1] == 1
