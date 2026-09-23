import torch
from helpers import D, assert_ulps, reference_layer, rel_err, rich_layer

from torch_type_nn import AndOr, and_or, readout
from torch_type_nn.functional import AndOrFunction, _ors, readout_grad


def test_readout_is_odd_and_compressive():
    u = torch.linspace(-5, 5, 101, dtype=D)
    assert torch.equal(readout(-u), -readout(u))
    assert readout(torch.zeros((), dtype=D)) == 0
    assert torch.all(readout(u).abs() <= u.abs())
    v = u.clone().requires_grad_(True)
    (torch.sign(v) * torch.log1p(v.abs())).sum().backward()
    nz = u != 0
    assert torch.allclose(v.grad[nz], readout_grad(u)[nz])


def test_forward_matches_the_definition():
    g = torch.Generator().manual_seed(11)
    layers = [rich_layer(3, 4, 3, g), rich_layer(4, 2, 2, g)]
    x = torch.tensor([0.3, -1.2, 0.8], dtype=D)
    cur, ref = x, x.tolist()
    for l in layers:
        ref = reference_layer(l, ref)
        cur = l(cur.unsqueeze(0)).squeeze(0)
    for a, b in zip(cur.tolist(), ref, strict=True):
        assert rel_err(a, b) < 1e-12


def test_batch_equals_per_sample_and_leading_dims():
    g = torch.Generator().manual_seed(1)
    l = rich_layer(5, 3, 2, g)
    x = torch.randn(7, 5, dtype=D, generator=g)
    zb = l(x)
    zs = torch.stack([l(x[i:i + 1])[0] for i in range(7)])
    assert torch.allclose(zb, zs, rtol=1e-13, atol=0)
    assert l(x.reshape(7, 1, 5)).shape == (7, 1, 3)


def test_gradcheck():
    g = torch.Generator().manual_seed(3)
    m, R, n = 3, 3, 4
    x = torch.randn(5, n, dtype=D, generator=g, requires_grad=True)
    w = torch.randn(m, R, n, dtype=D, generator=g, requires_grad=True)
    b = torch.randn(m, R, dtype=D, generator=g, requires_grad=True)
    a = (1.0 + 1.5 * torch.rand(m, R, dtype=D, generator=g)).requires_grad_(True)
    mask = torch.ones(m, R, dtype=torch.bool)
    assert torch.autograd.gradcheck(
        lambda *t: AndOrFunction.apply(*t, mask), (x, w, b, a), eps=1e-6, atol=1e-7)


def _stack_loss(layers, x, t):
    y = x
    for l in layers:
        y = l(y)
    return 0.5 * ((y - t) ** 2).mean()


def test_stack_gradients_match_central_differences():
    g = torch.Generator().manual_seed(99)
    h = 1e-6
    for _ in range(3):
        layers = [rich_layer(4, 3, 3, g), rich_layer(3, 3, 2, g), rich_layer(3, 3, 2, g)]
        x = (1.5 * (2 * torch.rand(1, 4, dtype=D, generator=g) - 1)).requires_grad_(True)
        t = 2 * torch.rand(1, 3, dtype=D, generator=g) - 1
        _stack_loss(layers, x, t).backward()
        with torch.no_grad():
            for j in range(4):
                xp, xm = x.clone(), x.clone()
                xp[0, j] += h
                xm[0, j] -= h
                fd = (_stack_loss(layers, xp, t) - _stack_loss(layers, xm, t)) / (2 * h)
                assert rel_err(x.grad[0, j], fd) < 1e-5
            for l in layers:
                for p, idx in ((l.bias, (1, 1)), (l.assembly, (0, 1)), (l.weight, (2, 0, 0))):
                    p0 = float(p[idx])
                    p[idx] = p0 + h
                    lp = _stack_loss(layers, x, t)
                    p[idx] = p0 - h
                    lm = _stack_loss(layers, x, t)
                    p[idx] = p0
                    assert rel_err(p.grad[idx], (lp - lm) / (2 * h)) < 1e-5


def test_zero_factor():
    g = torch.Generator().manual_seed(5)
    l0, l1 = rich_layer(2, 2, 2, g), rich_layer(2, 1, 2, g)
    x = torch.tensor([[0.4, -0.9]], dtype=D)
    t = torch.tensor([[0.3]], dtype=D)
    with torch.no_grad():
        l0.assembly[0, 0] = 1.0
        # Zero the factor through the layer's own matmul: a separate dot
        # product can round differently from the batched BLAS call (it does
        # on some builds), and then the factor is 1e-17, not 0.
        l0.bias[0, 0] = 0.0
        l0.bias[0, 0] = -_ors(x, l0.weight, l0.bias)[0, 0, 0]
        o = _ors(x, l0.weight, l0.bias)[0, 0, 0]
    assert float(o) == 0.0

    def loss():
        return _stack_loss([l0, l1], x, t)

    xg = x.clone().requires_grad_(True)
    _stack_loss([l0, l1], xg, t).backward()
    h = 1e-7
    with torch.no_grad():
        b0 = float(l0.bias[0, 0])
        l0.bias[0, 0] = b0 + h
        lp = loss()
        l0.bias[0, 0] = b0 - h
        lm = loss()
        l0.bias[0, 0] = b0
    assert rel_err(l0.bias.grad[0, 0], (lp - lm) / (2 * h)) < 1e-5, "a = 1: cofactor slope"
    assert torch.isfinite(xg.grad).all()
    for p in (*l0.parameters(), *l1.parameters()):
        p.grad = None
    with torch.no_grad():
        l0.assembly[0, 0] = 2.0
    loss().backward()
    assert l0.bias.grad[0, 0] == 0 and l0.assembly.grad[0, 0] == 0, "a > 1: flat at zero"
    assert all(torch.isfinite(p.grad).all() for p in l0.parameters())


def test_padding_is_an_exact_identity_and_gets_no_gradient():
    g = torch.Generator().manual_seed(2)
    l = rich_layer(4, 3, 2, g)
    x = torch.randn(6, 4, dtype=D, generator=g)
    before = l(x)
    l.add_or(1)                           # identity Or in a new slot column
    l.add_or(2)
    assert l.degree == 3
    assert_ulps(l(x), before)
    with torch.no_grad():
        l.mask[1, 2] = False              # make one of them padding
    l(x).sum().backward()
    assert l.weight.grad[1, 2].abs().sum() == 0
    assert l.bias.grad[1, 2] == 0 and l.assembly.grad[1, 2] == 0
    assert l.bias.grad[2, 2] != 0         # a live identity Or does learn


def test_functional_default_mask():
    g = torch.Generator().manual_seed(4)
    l = rich_layer(3, 2, 2, g)
    x = torch.randn(4, 3, dtype=D, generator=g)
    assert torch.equal(and_or(x, l.weight, l.bias, l.assembly), l(x))


def test_float32():
    l = AndOr(4, 3)
    assert l(torch.randn(2, 4)).dtype == torch.float32
