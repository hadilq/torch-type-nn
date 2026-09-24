"""The type-nn adapter: :class:`~torch_type_nn.network.TypeNN` as a :class:`Scalable`.

Addresses:

* ``Item(WIDTH, (i, k))``: output coordinate ``k`` of layer ``i`` (the unit
  and the column of layer ``i + 1`` that reads it); sites are junctions ``i``.
* ``Item(DEGREE, (i, k, r))``: Or slot ``r`` of unit ``k`` of layer ``i``;
  sites are units ``(i, k)``.
* ``Item(DEPTH, (i,))``: layer ``i``; sites are gaps ``g`` (in front of layer
  ``g``), counted in the stack without the depth probe.

Width grows on *every* junction (``width_through_depth_probe=True``, the
default): a width probe on a junction that crosses the depth probe passes
through the probe layer on a carrier Or (w = e_k, b = 0, a = 1) and the real
consumer beyond it reads it with weights of exactly 0. The C reference skips
such junctions, which leaves a network born with depth <= 2 no width site
while a depth probe exists (see docs/DIFFERENCES.md);
``width_through_depth_probe=False`` reproduces it.

The numerical content is ``type_nn_scale.c``: distances from the identity,
exact ablations, affine folds for depth edits and the exact parameter count
of a trial model (see docs/DIFFERENCES.md).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..layer import AndOr, uniform
from ..network import TypeNN
from ..protocol import DEGREE, DEPTH, WIDTH, EditContext, Item, Trial

__all__ = ["TypeNNAdapter", "dev_or", "dev_column", "dev_layer"]


# ------------------------------------------------------- distances from identity

def dev_or(layer: AndOr, k: int, r: int) -> float:
    """RMS distance of an Or from the unit identity (w = 0, b = 1, a = 1)."""
    w, b, a = (layer.weight.detach()[k, r], layer.bias.detach()[k, r],
               layer.assembly.detach()[k, r])
    s = float((w * w).sum() + (b - 1) ** 2 + (a - 1) ** 2)
    return math.sqrt(s / (layer.in_features + 2))


def dev_column(consumer: AndOr, j: int) -> float:
    """RMS of the weights with which ``consumer`` reads coordinate ``j``."""
    live = consumer.mask
    c = int(live.sum())
    if not c:
        return 0.0
    w = consumer.weight.detach()[..., j][live]
    return math.sqrt(float((w * w).sum()) / c)


def _sq_one(layer: AndOr) -> torch.Tensor:
    w, b, a = layer.weight.detach(), layer.bias.detach(), layer.assembly.detach()
    return (w * w).sum(-1) + (b - 1) ** 2 + (a - 1) ** 2                # (m, R)


def _sq_carrier(layer: AndOr) -> torch.Tensor:
    # carrier of coordinate k: w = e_k, b = 0, a = 1 (the Or equals x_k)
    m = layer.out_features
    eye = torch.eye(m, layer.in_features, dtype=layer.weight.dtype,
                    device=layer.weight.device).unsqueeze(1)              # (m, 1, n)
    d = layer.weight.detach() - eye
    return (d * d).sum(-1) + layer.bias.detach() ** 2 + (layer.assembly.detach() - 1) ** 2


def dev_layer(layer: AndOr) -> float:
    """RMS distance from the identity layer; inf if the layer is not square."""
    if layer.in_features != layer.out_features:
        return math.inf
    live = layer.mask
    one, car = _sq_one(layer), _sq_carrier(layer)
    all_one = torch.where(live, one, torch.zeros_like(one)).sum(1, keepdim=True)
    v = torch.where(live, all_one - one + car, torch.full_like(one, math.inf))
    s = float(v.min(1).values.sum()) if layer.degree else math.inf
    c = int(live.sum()) * (layer.in_features + 2)
    return math.sqrt(s / c) if c else math.inf


# ------------------------------------------------------- exact ablations

def _snap(layer: AndOr, *names: str) -> dict:
    return {n: getattr(layer, n).detach().clone() for n in names}


@torch.no_grad()
def _restore(layer: AndOr, snap: dict) -> None:
    for n, v in snap.items():
        getattr(layer, n).copy_(v)


@torch.no_grad()
def _ablate_or(layer: AndOr, k: int, r: int):
    s = (layer.weight[k, r].clone(), layer.bias[k, r].clone(), layer.assembly[k, r].clone())
    layer.weight[k, r] = 0.0
    layer.bias[k, r] = 1.0
    layer.assembly[k, r] = 1.0

    def undo() -> None:
        with torch.no_grad():
            layer.weight[k, r], layer.bias[k, r], layer.assembly[k, r] = s
    return undo


@torch.no_grad()
def _ablate_col(consumer: AndOr, j: int):
    s = consumer.weight[..., j].clone()
    consumer.weight[..., j] = 0.0

    def undo() -> None:
        with torch.no_grad():
            consumer.weight[..., j] = s
    return undo


@torch.no_grad()
def _ablate_layer(layer: AndOr):
    """Reset a square layer to its identity: per unit the live Or nearest e_k
    becomes the carrier, every other Or the unit identity."""
    snap = _snap(layer, "weight", "bias", "assembly")
    live = layer.mask
    d = torch.where(live, _sq_carrier(layer) - _sq_one(layer),
                    torch.full_like(layer.bias, math.inf))
    best = d.argmin(1)                                                   # (m,)
    layer.weight.masked_fill_(live.unsqueeze(-1), 0.0)
    layer.bias.masked_fill_(live, 1.0)
    layer.assembly.fill_(1.0)
    ks = torch.arange(layer.out_features, device=best.device)
    has = live[ks, best]
    ks, best = ks[has], best[has]
    layer.weight[ks, best, ks] = 1.0
    layer.bias[ks, best] = 0.0
    return lambda: _restore(layer, snap)


# ------------------------------------------------------- folds

def fold_from_stats(at: AndOr, u_to_x: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """A type-identity layer emits u = F(x), which is x only to first order.
    Insert (u_to_x): x ~ alpha u + beta. Remove: u ~ alpha x + beta.
    Measured on the epoch that just ran."""
    n = at.in_features
    one, zero = at.weight.new_ones(n), at.weight.new_zeros(n)
    ns = float(at.stat_ns)
    if ns < 2:
        return one, zero
    mx, mu = at.stat_sx / ns, at.stat_su / ns
    vxx = at.stat_sxx / ns - mx * mx
    vuu = at.stat_suu / ns - mu * mu
    vxu = at.stat_sxu / ns - mx * mu
    if u_to_x:
        alpha = torch.where(vuu > 0, vxu / torch.where(vuu > 0, vuu, one), one)
        beta = mx - alpha * mu
    else:
        alpha = torch.where(vxx > 0, vxu / torch.where(vxx > 0, vxx, one), one)
        beta = mu - alpha * mx
    bad = ~(alpha > 0) | ~torch.isfinite(alpha)
    return torch.where(bad, one, alpha), torch.where(bad, zero, beta)


@torch.no_grad()
def fold_apply(c: AndOr, alpha: torch.Tensor, beta: torch.Tensor) -> None:
    J = min(alpha.shape[0], c.in_features)
    a, b = alpha[:J], beta[:J]
    c.bias += (c.weight[..., :J] * b).sum(-1)
    c.weight[..., :J] *= a
    c.adam_m_weight[..., :J] /= a
    c.adam_v_weight[..., :J] /= a * a


# ------------------------------------------------------- the adapter

class TypeNNAdapter:
    """:class:`TypeNN` under the :class:`~torch_type_nn.protocol.Scalable` protocol."""

    def __init__(self, model: TypeNN, *, width_through_depth_probe: bool = True) -> None:
        self.model = model
        self.through = bool(width_through_depth_probe)
        self._sink: list | None = None
        self._tracking = False

    @property
    def layers(self):
        return self.model.layers

    # ------------------------------------------------------------ bookkeeping

    def num_params(self) -> int:
        return self.model.num_params()

    def set_tracking(self, on: bool) -> None:
        self._tracking = on
        for layer in self.layers:
            layer.track_stats = on

    def reset_stats(self) -> None:
        for layer in self.layers:
            layer.reset_stats()

    def set_edit_sink(self, sink: list | None) -> None:
        self._sink = sink
        for layer in self.layers:
            layer.edit_sink = sink

    def finalize(self) -> None:
        for layer in self.layers:
            layer.compact()

    # ----------------------------------------------------------------- probes

    def _probe_layer(self) -> int | None:
        for i, layer in enumerate(self.layers):
            if layer.probe:
                return i
        return None

    @staticmethod
    def _probe_unit(layer: AndOr) -> int | None:
        k = layer.unit_probe.nonzero()
        return int(k[0]) if len(k) else None

    @staticmethod
    def _probe_or(layer: AndOr, k: int) -> int | None:
        r = (layer.or_probe[k] & layer.mask[k]).nonzero()
        return int(r[0]) if len(r) else None

    def _crosses_probe(self, i: int) -> bool:
        return self.through and self.layers[i + 1].probe

    def _consumer(self, i: int) -> AndOr:
        """The layer that reads the output coordinates of layer i."""
        return self.layers[i + 2] if self._crosses_probe(i) else self.layers[i + 1]

    def probe_sites(self, axis: str) -> Sequence:
        L = self.layers
        if axis == WIDTH:
            if self.through:   # every non-probe producer; the probe passes units on
                return [i for i in range(len(L) - 1) if not L[i].probe]
            # the reference rule: junctions not touching the depth probe
            return [i for i in range(len(L) - 1) if not (L[i].probe or L[i + 1].probe)]
        if axis == DEGREE:
            return [(i, k) for i, l in enumerate(L) for k in range(l.out_features)]
        return []

    def probe_at(self, axis: str, site=None) -> Item | None:
        if axis == WIDTH:
            k = self._probe_unit(self.layers[site])
            return None if k is None else Item(WIDTH, (site, k))
        if axis == DEGREE:
            i, k = site
            r = self._probe_or(self.layers[i], k)
            return None if r is None else Item(DEGREE, (i, k, r))
        p = self._probe_layer()
        return None if p is None else Item(DEPTH, (p,))

    def add_probe(self, axis: str, site, ctx: EditContext) -> None:
        if axis == WIDTH:
            self._add_width_probe(site, ctx)
        elif axis == DEGREE:
            i, k = site
            layer = self.layers[i]
            n = layer.in_features
            w = uniform((n,), ctx.noise, ctx.generator, layer.weight)
            b = 1.0 + float(uniform((), ctx.noise, ctx.generator, layer.bias))
            layer.add_or(k, w, b, 1.0, probe=True, born=ctx.step)
        else:
            self._insert_depth_probe(site, ctx)

    def promote(self, item: Item) -> None:
        if item.axis == WIDTH:
            i, k = item.address
            self.layers[i].unit_probe[k] = False
        elif item.axis == DEGREE:
            i, k, r = item.address
            self.layers[i].or_probe[k, r] = False
        else:
            self.layers[item.address[0]].probe = False

    def born(self, item: Item) -> int:
        if item.axis == WIDTH:
            i, k = item.address
            return int(self.layers[i].unit_born[k])
        if item.axis == DEGREE:
            i, k, r = item.address
            return int(self.layers[i].or_born[k, r])
        return self.layers[item.address[0]].born

    def displacement(self, item: Item) -> float:
        if item.axis == WIDTH:
            i, k = item.address
            return dev_column(self._consumer(i), k)
        if item.axis == DEGREE:
            i, k, r = item.address
            return dev_or(self.layers[i], k, r)
        return dev_layer(self.layers[item.address[0]])

    def best_site(self, axis: str, moving: Item | None = None):
        """Gap with the largest mean |dL/dx| over the epoch, counted in the
        stack without the depth probe. The gap behind the last layer has no
        consumer to absorb a fold and is not a candidate."""
        probe = None if moving is None else moving.address[0]

        def loud(layer: AndOr) -> float:
            ns = float(layer.stat_ns)
            return float(layer.stat_gin) / ns if ns else 0.0

        best_g, best, g = 0, -1.0, 0
        for i, layer in enumerate(self.layers):
            if i == probe:
                continue
            v = loud(layer)
            if probe is not None and i == probe + 1:
                v = max(v, loud(self.layers[probe]))
            if v > best:
                best, best_g = v, g
            g += 1
        return best_g

    def site_of(self, item: Item):
        return item.address[0]

    def remove_probe(self, item: Item) -> None:
        self._remove_identity_layer(item.address[0])

    # ------------------------------------------------------ structural edits

    def _add_width_probe(self, i: int, ctx: EditContext) -> None:
        """The producing layer grows a new, trained output unit; the consuming
        layer meets it with weights born at exactly 0."""
        p, c = self.layers[i], self._consumer(i)
        k = p.add_unit()
        bound = 1.0 / math.sqrt(max(p.in_features, 1))
        w = uniform((p.in_features,), bound, ctx.generator, p.weight)
        b = float(uniform((), bound, ctx.generator, p.bias))
        p.add_or(k, w, b, 1.0, born=ctx.step)
        p.unit_probe[k] = True
        p.unit_born[k] = ctx.step
        if self._crosses_probe(i):
            # the depth probe carries it: a new input and a unit whose one Or
            # is the carrier of that input (w = e_k, b = 0, a = 1)
            P = self.layers[i + 1]
            P.add_input()
            kp = P.add_unit()
            e = P.weight.new_zeros(P.in_features)
            e[kp] = 1.0
            P.add_or(kp, e, 0.0, 1.0, born=ctx.step)
        c.add_input()

    def _input_means(self, i: int) -> torch.Tensor | None:
        """E[x] of the consumer of junction i over the epoch (None without data)."""
        c = self._consumer(i)
        ns = float(c.stat_ns)
        return (c.stat_sx / ns).clone() if ns > 0 else None

    @torch.no_grad()
    def _drop_coordinate(self, i: int, k: int, mean: float | None = None) -> None:
        """Drop output coordinate k of layer i; the consumer keeps the mean of
        what it contributed (b += w E[x_k]). ``mean`` is E[x_k] taken before
        any other drop on the junction (a drop resets the consumer's statistics)."""
        p, c = self.layers[i], self._consumer(i)
        if mean is None:
            means = self._input_means(i)
            mean = None if means is None else means[k]
        if mean is not None:
            c.bias += c.weight[..., k] * mean
        if self._crosses_probe(i):
            P = self.layers[i + 1]
            P.drop_unit(k)
            P.drop_input(k)
        p.drop_unit(k)
        c.drop_input(k)

    @torch.no_grad()
    def _insert_depth_probe(self, g: int, ctx: EditContext) -> None:
        """Identity layer of width d at gap g (in front of layer g)."""
        layers = self.layers
        c = layers[g]
        alpha, beta = fold_from_stats(c, u_to_x=True)
        # A width probe on the junction in front of c would be a coordinate
        # the identity layer has to carry; the reference retires it first.
        # (Passing units through the probe, the identity carries it.)
        if g > 0 and not layers[g - 1].probe and not self.through:
            k = self._probe_unit(layers[g - 1])
            if k is not None:
                self._drop_coordinate(g - 1, k)
                keep = [j for j in range(alpha.shape[0]) if j != k]
                alpha, beta = alpha[keep], beta[keep]
        d = c.in_features
        l = AndOr(d, d, 1, device=c.weight.device, dtype=c.weight.dtype)
        w = uniform((d, 1, d), ctx.noise, ctx.generator, l.weight)
        w[:, 0, :] += torch.eye(d, dtype=w.dtype, device=w.device)
        l.weight.copy_(w)
        l.bias.copy_(uniform((d, 1), ctx.noise, ctx.generator, l.bias))
        l.assembly.fill_(1.0)
        l.or_born.fill_(ctx.step)
        l.probe, l.born, l.track_stats = True, ctx.step, self._tracking
        l.edit_sink = self._sink
        fold_apply(c, alpha, beta)
        c.reset_stats()
        self.model.insert_layer(g, l)

    def _remove_identity_layer(self, i: int) -> None:
        """Remove a (near-)identity layer; layer i+1 absorbs it."""
        l, c = self.layers[i], self.layers[i + 1]
        alpha, beta = fold_from_stats(l, u_to_x=False)
        fold_apply(c, alpha, beta)
        c.reset_stats()
        self.model.remove_layer(i)

    # -------------------------------------------------- measurement / pruning

    def ablate(self, item: Item):
        if item.axis == WIDTH:
            i, k = item.address
            return _ablate_col(self._consumer(i), k)
        if item.axis == DEGREE:
            i, k, r = item.address
            return _ablate_or(self.layers[i], k, r)
        layer = self.layers[item.address[0]]
        if layer.in_features != layer.out_features:
            return None
        return _ablate_layer(layer)

    def cost(self, item: Item) -> int:
        if item.axis == WIDTH:
            i, j = item.address
            p, c = self.layers[i], self._consumer(i)
            cost = int(p.mask[j].sum()) * (p.in_features + 2) + int(c.mask.sum())
            if self._crosses_probe(i):      # its carrier Or and its probe column
                P = self.layers[i + 1]
                cost += (P.in_features + 2) + int(P.mask.sum())
            return cost
        if item.axis == DEGREE:
            return self.layers[item.address[0]].in_features + 2
        return self.layers[item.address[0]].num_params()

    def trial_remove(self, item: Item) -> Trial | None:
        """Remove square layer i; layer i+1 absorbs the fold."""
        i = item.address[0]
        l, c = self.layers[i], self.layers[i + 1]
        if l.in_features != l.out_features:
            return None
        snap = _snap(c, "weight", "bias", "adam_m_weight", "adam_v_weight")
        alpha, beta = fold_from_stats(l, u_to_x=False)
        fold_apply(c, alpha, beta)
        self.model.remove_layer(i)

        def undo() -> None:
            self.model.insert_layer(i, l)
            _restore(c, snap)

        return Trial(undo=undo, commit=c.reset_stats)

    def removal_order(self, axis: str) -> Sequence[Item]:
        if axis == DEPTH:     # never the output layer
            return [Item(DEPTH, (i,)) for i in range(len(self.layers) - 1)]
        return []

    def prune_candidates(self) -> Sequence[Item]:
        out = []
        L = self.layers
        for i, l in enumerate(L):
            out += [Item(DEGREE, (i, k, r)) for k, r in l.mask.nonzero().tolist()]
            if i + 1 < len(L):
                out += [Item(WIDTH, (i, k)) for k in range(l.out_features)]
        return out

    @staticmethod
    def _gone(removed: Sequence[Item]):
        units = {it.address for it in removed if it.axis == WIDTH}
        ors = {it.address for it in removed if it.axis == DEGREE}
        return units, ors

    def can_remove(self, item: Item, removed: Sequence[Item]) -> bool:
        units, ors = self._gone(removed)
        i, k = item.address[:2]
        if (i, k) in units:
            return False
        if item.axis == DEGREE:        # keep at least one Or per unit
            left = int(self.layers[i].mask[k].sum()) - sum(
                1 for a in ors if a[0] == i and a[1] == k)
            return left > 1
        left = self.layers[i].out_features - sum(1 for a in units if a[0] == i)
        return left > 1                # keep at least one coordinate per junction

    def params_without(self, removed: Sequence[Item]) -> int:
        """(live Ors) x (live inputs + 2) per surviving unit. (The C reference
        sums a per-item tally that can count an Or twice; see DIFFERENCES.md.)"""
        units, ors = self._gone(removed)
        total, n_in = 0, self.layers[0].in_features
        for i, l in enumerate(self.layers):
            n_out = 0
            for k, n_or in enumerate(l.num_ors().tolist()):
                if (i, k) in units:
                    continue
                n_out += 1
                gone = sum(1 for a in ors if a[0] == i and a[1] == k)
                total += (n_or - gone) * (n_in + 2)
            n_in = n_out
        return total

    def commit_removals(self, removed: Sequence[Item],
                        axes: Sequence[str] = (WIDTH, DEGREE)) -> dict[str, int]:
        """Remove the chosen items, then clear the probe flags of what stays, on
        the given axes only. Under the BIC rule the items were ablated (an
        identity: exact); under the threshold rule they sit inside the band."""
        units, ors = self._gone(removed)
        c = {"or_add": 0, "or_drop": 0, "and_add": 0, "and_drop": 0}
        layers = self.layers
        do_width, do_degree = WIDTH in axes, DEGREE in axes
        means = {i: self._input_means(i) for i in range(len(layers) - 1)
                 if any(a[0] == i for a in units)}
        for i in reversed(range(len(layers))):
            l = layers[i]
            for k in reversed(range(l.out_features)):
                gone = (i, k) in units
                for r in reversed(range(l.degree if do_degree else 0)):
                    if not bool(l.mask[k, r]):
                        continue
                    was_probe = bool(l.or_probe[k, r])
                    if (i, k, r) in ors and not gone:
                        l.drop_or(k, r)
                        if not was_probe:
                            c["and_drop"] += 1
                    elif was_probe and not gone:
                        l.or_probe[k, r] = False
                        c["and_add"] += 1
                if do_width and i + 1 < len(layers):
                    was_probe = bool(l.unit_probe[k])
                    if gone:
                        m = means.get(i)
                        self._drop_coordinate(i, k, None if m is None else m[k])
                        if not was_probe:
                            c["or_drop"] += 1
                    elif was_probe:
                        l.unit_probe[k] = False
                        c["or_add"] += 1
        for l in layers:
            l.compact()
        return c
