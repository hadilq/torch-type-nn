import math

import pytest
import torch
from helpers import D, assert_ulps, rich_layer

from torch_type_nn import AndOr


def test_init_law():
    g = torch.Generator().manual_seed(0)
    l = AndOr(16, 5, 2, generator=g, dtype=D)
    k = 1 / math.sqrt(16)
    assert l.weight.abs().max() <= k and l.bias.abs().max() <= k
    assert torch.all(l.assembly == 1) and bool(l.mask.all())
    assert l.num_params() == 5 * 2 * (16 + 2)


def test_add_input_preserves_function_and_drop_undoes_it():
    g = torch.Generator().manual_seed(1)
    l = rich_layer(3, 4, 2, g)
    x = torch.randn(8, 3, dtype=D, generator=g)
    y = l(x)
    l.add_input()
    x4 = torch.cat([x, torch.randn(8, 1, dtype=D, generator=g)], 1)
    assert l.in_features == 4 and l.weight.shape == (4, 2, 4)
    assert_ulps(l(x4), y)
    l.drop_input(3)
    assert_ulps(l(x), y)


def test_or_and_unit_edits():
    g = torch.Generator().manual_seed(2)
    l = rich_layer(3, 2, 1, g)
    x = torch.randn(5, 3, dtype=D, generator=g)
    y = l(x)
    r = l.add_or(0, probe=True, born=7)
    assert (r, l.degree) == (1, 2) and bool(l.or_probe[0, 1]) and int(l.or_born[0, 1]) == 7
    assert_ulps(l(x), y)                  # identity Or: exact up to BLAS rounding
    l.drop_or(0, r)
    l.compact()
    assert l.degree == 1
    assert_ulps(l(x), y)
    k = l.add_unit()
    assert k == 2 and l.out_features == 3 and int(l.num_ors()[2]) == 0
    assert torch.allclose(l(x)[:, 2], torch.full((5,), math.log(2), dtype=D))
    l.add_or(k, torch.ones(3, dtype=D), 0.5)
    assert l.num_params() == 3 * 5
    l.drop_unit(0)
    assert l.out_features == 2
    assert_ulps(l(x)[:, 0], y[:, 1])


def test_project():
    l = AndOr(2, 2, 2, dtype=D)
    with torch.no_grad():
        l.assembly[0, 0] = 0.5
        l.mask[1, 1] = False
        l.weight[1, 1] = 3.0
    l.project_()
    assert l.assembly[0, 0] == 1.0
    assert l.weight[1, 1].abs().sum() == 0 and l.bias[1, 1] == 1


def test_state_dict_roundtrip_across_structures():
    g = torch.Generator().manual_seed(3)
    a = rich_layer(3, 2, 1, g)
    a.add_or(0, torch.full((3,), 0.1, dtype=D), 0.9, 1.5)
    a.add_input()
    a.add_unit()
    a.probe, a.born = True, 42
    b = AndOr(1, 1, 1, dtype=D)
    b.load_state_dict(a.state_dict())
    assert (b.in_features, b.out_features, b.degree) == (4, 3, 2)
    assert b.probe and b.born == 42
    x = torch.randn(4, 4, dtype=D, generator=g)
    assert torch.equal(a(x), b(x))


def test_input_statistics():
    g = torch.Generator().manual_seed(4)
    l = rich_layer(3, 2, 2, g)
    l.track_stats = True
    x = torch.randn(10, 3, dtype=D, generator=g)
    l(x).sum().backward()
    assert int(l.stat_ns) == 10
    assert torch.allclose(l.stat_sx, x.sum(0))
    assert float(l.stat_gin) > 0
    with torch.no_grad():
        l(x)
    assert int(l.stat_ns) == 10, "no statistics outside training passes"
    l.reset_stats()
    assert int(l.stat_ns) == 0 and float(l.stat_gin) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda():
    l = rich_layer(3, 2, 2, torch.Generator().manual_seed(0)).cuda()
    l.add_or(0)
    l.add_input()
    y = l(torch.randn(4, 4, dtype=D, device="cuda"))
    y.sum().backward()
    assert y.is_cuda and l.weight.grad.is_cuda
