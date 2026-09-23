"""Shared test helpers: an independent pow-based reference and rich layers."""

import math

import torch

from torch_type_nn import AndOr

D = torch.float64


def rich_layer(n, m, degree, gen, a_max=2.5):
    """A layer with several Ors per unit and assembly indices above 1."""
    l = AndOr(n, m, degree, generator=gen, dtype=D)
    with torch.no_grad():
        l.weight.uniform_(-1.0, 1.0, generator=gen)
        l.bias.uniform_(-1.0, 1.0, generator=gen)
        l.assembly.uniform_(1.0, a_max, generator=gen)
    return l


def reference_layer(layer, x):
    """z for one sample by the definition: A = prod sign(o)|o|^a, z = F(A)."""
    out = []
    layer = _Detached(layer)
    for k in range(layer.out_features):
        A = 1.0
        for r in range(layer.degree):
            if not bool(layer.mask[k, r]):
                continue
            o = float(layer.bias[k, r]) + sum(
                float(layer.weight[k, r, j]) * float(x[j]) for j in range(layer.in_features))
            A *= math.copysign(abs(o) ** float(layer.assembly[k, r]), o)
        out.append(math.copysign(math.log1p(abs(A)), A))
    return out


def assert_ulps(a, b, ulps=8):
    """Equal up to a few units in the last place.

    Structural edits are exact in real arithmetic, and the C reference
    reproduces them bit for bit because it sums in a fixed order. A batched
    matmul does not promise that: a different shape (one more input column,
    one more Or slot) may take a different BLAS reduction or FMA path, which
    moves the last bit on some builds.
    """
    a, b = a.detach(), b.detach()
    tol = ulps * torch.finfo(a.dtype).eps * torch.maximum(a.abs(), b.abs()).clamp(min=1.0)
    diff = (a - b).abs()
    assert bool((diff <= tol).all()), f"max diff {float(diff.max()):.3e}"


def rel_err(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(1.0, abs(a), abs(b))


class _Detached:
    def __init__(self, l):
        self.in_features, self.out_features, self.degree = l.in_features, l.out_features, l.degree
        self.mask, self.weight = l.mask, l.weight.detach()
        self.bias, self.assembly = l.bias.detach(), l.assembly.detach()
