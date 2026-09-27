"""CudaTypeNN as a Scalable: the three-axis type-nn rule on a ragged store."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..protocol import DEGREE, DEPTH, WIDTH, EditContext, Item, Trial
from .model import PackedLayer, pack_layer

__all__ = ["CudaAdapter"]


def _noise(shape, amp, gen, ref):
    """U(-amp, amp) on the generator's device, then moved onto ``ref``."""
    dev = gen.device if gen is not None else ref.device
    t = torch.empty(shape, dtype=ref.dtype, device=dev)
    t.uniform_(-amp, amp, generator=gen)
    return t.to(ref.device)


def _dev_or(layer: PackedLayer, r: int) -> float:
    w, b, a = layer.W[r], layer.b[r], layer.a[r]
    s = float((w * w).sum() + (b - 1) ** 2 + (a - 1) ** 2)
    return math.sqrt(s / (layer.n_in + 2))


def _dev_column(consumer: PackedLayer, j: int) -> float:
    if consumer.n_or == 0:
        return 0.0
    w = consumer.W[:, j]
    return math.sqrt(float((w * w).sum()) / consumer.n_or)


def _dev_layer(layer: PackedLayer) -> float:
    if layer.n_in != layer.n_out:
        return math.inf
    p = layer.ptr.tolist()
    s = 0.0
    c = 0
    eye = torch.eye(layer.n_in, dtype=layer.W.dtype, device=layer.W.device)
    for k in range(layer.n_out):
        best = math.inf
        for r in range(p[k], p[k + 1]):
            d = layer.W[r] - eye[k]
            v = float((d * d).sum() + layer.b[r] ** 2 + (layer.a[r] - 1) ** 2)
            if v < best:
                best = v
        s += best
        c += layer.n_in + 2
    return math.sqrt(s / c) if c else math.inf


def _fold_from_stats(layer: PackedLayer, u_to_x: bool):
    n = layer.n_in
    one = torch.ones(n, dtype=torch.float64, device=layer.W.device)
    zero = torch.zeros(n, dtype=torch.float64, device=layer.W.device)
    ns = float(layer.stat_ns)
    if ns < 2 or layer.sx is None:
        return one, zero
    mx, mu = layer.sx / ns, layer.su / ns
    vxx = layer.sxx / ns - mx * mx
    vuu = layer.suu / ns - mu * mu
    vxu = layer.sxu / ns - mx * mu
    if u_to_x:
        alpha = torch.where(vuu > 0, vxu / torch.where(vuu > 0, vuu, one), one)
        beta = mx - alpha * mu
    else:
        alpha = torch.where(vxx > 0, vxu / torch.where(vxx > 0, vxx, one), one)
        beta = mu - alpha * mx
    bad = ~(alpha > 0) | ~torch.isfinite(alpha)
    return torch.where(bad, one, alpha), torch.where(bad, zero, beta)


def _fold_apply(c: PackedLayer, alpha: torch.Tensor, beta: torch.Tensor) -> None:
    j = min(int(alpha.shape[0]), c.n_in)
    a, b = alpha[:j], beta[:j]
    c.b = c.b + (c.W[:, :j] * b).sum(-1)
    c.W[:, :j] = c.W[:, :j] * a
    c.mW[:, :j] = c.mW[:, :j] / a
    c.vW[:, :j] = c.vW[:, :j] / (a * a)


class CudaAdapter:
    """Ragged CudaTypeNN under the Scalable protocol."""

    def __init__(self, model, *, width_through_depth_probe: bool = True) -> None:
        self.model = model
        self.through = bool(width_through_depth_probe)
        self._sink = None
        self._tracking = False

    @property
    def layers(self) -> list[PackedLayer]:
        return self.model.layers

    def num_params(self) -> int:
        return self.model.num_params()

    def set_tracking(self, on: bool) -> None:
        self._tracking = on
        for layer in self.layers:
            layer.track_stats = on

    def reset_stats(self) -> None:
        for layer in self.layers:
            layer.reset_stats()

    def set_edit_sink(self, sink) -> None:
        self._sink = sink

    def finalize(self) -> None:
        return

    def _probe_layer(self) -> int | None:
        for i, layer in enumerate(self.layers):
            if layer.layer_probe:
                return i
        return None

    def _probe_unit(self, layer: PackedLayer) -> int | None:
        layer.ready()
        hit = (layer.unit_probe != 0).nonzero()
        return int(hit[0]) if len(hit) else None

    def _probe_or(self, layer: PackedLayer, k: int) -> int | None:
        p = layer.ptr.tolist()
        for r in range(p[k], p[k + 1]):
            if int(layer.probe[r]):
                return r - p[k]
        return None

    def _crosses_probe(self, i: int) -> bool:
        return self.through and i + 1 < len(self.layers) and self.layers[i + 1].layer_probe

    def _consumer(self, i: int) -> PackedLayer:
        return self.layers[i + 2] if self._crosses_probe(i) else self.layers[i + 1]

    def probe_sites(self, axis: str) -> Sequence:
        L = self.layers
        if axis == WIDTH:
            if self.through:
                return [i for i in range(len(L) - 1) if not L[i].layer_probe]
            return [i for i in range(len(L) - 1)
                    if not (L[i].layer_probe or L[i + 1].layer_probe)]
        if axis == DEGREE:
            return [(i, k) for i, l in enumerate(L) for k in range(l.n_out)]
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
            w = _noise((layer.n_in,), ctx.noise, ctx.generator, layer.W)
            b = 1.0 + float(_noise((), ctx.noise, ctx.generator, layer.b))
            self._add_or(layer, k, w, b, 1.0, probe=1, born=ctx.step)
        else:
            self._insert_depth_probe(site, ctx)

    def promote(self, item: Item) -> None:
        if item.axis == WIDTH:
            i, k = item.address
            self.layers[i].ready()
            self.layers[i].unit_probe[k] = 0
        elif item.axis == DEGREE:
            i, k, r = item.address
            p = int(self.layers[i].ptr[k])
            self.layers[i].probe[p + r] = 0
        else:
            self.layers[item.address[0]].layer_probe = 0

    def born(self, item: Item) -> int:
        if item.axis == WIDTH:
            i, k = item.address
            self.layers[i].ready()
            return int(self.layers[i].unit_born[k])
        if item.axis == DEGREE:
            i, k, r = item.address
            p = int(self.layers[i].ptr[k])
            return int(self.layers[i].born[p + r])
        return int(self.layers[item.address[0]].layer_born)

    def displacement(self, item: Item) -> float:
        if item.axis == WIDTH:
            i, k = item.address
            return _dev_column(self._consumer(i), k)
        if item.axis == DEGREE:
            i, k, r = item.address
            p = int(self.layers[i].ptr[k])
            return _dev_or(self.layers[i], p + r)
        return _dev_layer(self.layers[item.address[0]])

    def best_site(self, axis: str, moving: Item | None = None):
        probe = None if moving is None else moving.address[0]

        def loud(layer: PackedLayer) -> float:
            ns = float(layer.stat_ns)
            return layer.gin / ns if ns else 0.0

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

    def _add_or(self, layer: PackedLayer, k: int, w, b, a, probe, born) -> None:
        rows = _unpack(layer)
        ww = w.detach().cpu() if torch.is_tensor(w) else torch.as_tensor(w)
        rows[k].append((ww, float(b), float(a), 0, int(born), int(probe)))
        new = pack_layer(layer.n_in, rows, layer.W.device)
        _copy_meta(layer, new)
        # restore moments for old rows
        self._replace(layer, new)

    def _replace(self, old: PackedLayer, new: PackedLayer) -> None:
        i = next(j for j, layer in enumerate(self.layers) if layer is old)
        new.layer_probe = old.layer_probe
        new.layer_born = old.layer_born
        new.track_stats = old.track_stats
        new.gin = old.gin
        new.stat_ns = old.stat_ns
        if old.sx is not None and old.sx.numel() == new.n_in:
            new.sx, new.sxx = old.sx, old.sxx
            new.su, new.suu, new.sxu = old.su, old.suu, old.sxu
        if old.unit_probe is not None and old.unit_probe.numel() == new.n_out:
            new.unit_probe = old.unit_probe
            new.unit_born = old.unit_born
        self.layers[i] = new.ready()

    def _add_width_probe(self, i: int, ctx: EditContext) -> None:
        p, c = self.layers[i], self._consumer(i)
        bound = 1.0 / math.sqrt(max(p.n_in, 1))
        w = _noise((p.n_in,), bound, ctx.generator, p.W)
        b = float(_noise((), bound, ctx.generator, p.b))
        rows = _unpack(p)
        rows.append([(w.detach().cpu(), b, 1.0, 0, ctx.step, 0)])
        new = pack_layer(p.n_in, rows, p.W.device)
        new.ready()
        if p.unit_probe is not None:
            new.unit_probe = torch.cat([p.unit_probe,
                                        torch.ones(1, dtype=torch.int64,
                                                   device=p.W.device)])
            new.unit_born = torch.cat([p.unit_born,
                                       torch.tensor([ctx.step], dtype=torch.int64,
                                                    device=p.W.device)])
        else:
            new.unit_probe[-1] = 1
            new.unit_born[-1] = ctx.step
        self._replace(p, new)
        if self._crosses_probe(i):
            P = self.layers[i + 1]
            self._add_input(P)
            self._add_carrier_unit(P, ctx.step)
        self._add_input(c)

    def _add_carrier_unit(self, layer: PackedLayer, born: int) -> None:
        e = torch.zeros(layer.n_in)
        e[-1] = 1.0
        rows = _unpack(layer)
        rows.append([(e, 0.0, 1.0, 0, born, 0)])
        new = pack_layer(layer.n_in, rows, layer.W.device)
        self._replace(layer, new)

    def _add_input(self, layer: PackedLayer) -> None:
        z = torch.zeros(layer.n_or, 1, dtype=layer.W.dtype, device=layer.W.device)
        layer.W = torch.cat([layer.W, z], 1)
        layer.mW = torch.cat([layer.mW, z.clone()], 1)
        layer.vW = torch.cat([layer.vW, z.clone()], 1)
        layer.n_in += 1
        layer.ready()

    def _drop_or(self, i: int, k: int, r: int) -> None:
        rows = _unpack(self.layers[i])
        del rows[k][r]
        if not rows[k]:
            return
        new = pack_layer(self.layers[i].n_in, rows, self.layers[i].W.device)
        self._replace(self.layers[i], new)

    def _drop_coordinate(self, i: int, k: int, mean: float | None = None) -> None:
        p, c = self.layers[i], self._consumer(i)
        if mean is None and c.stat_ns > 0 and c.sx is not None:
            mean = float(c.sx[k] / c.stat_ns)
        if mean is not None and k < c.n_in:
            c.b = c.b + c.W[:, k] * mean
        if self._crosses_probe(i):
            P = self.layers[i + 1]
            self._drop_unit(P, k)
            self._drop_input(P, k)
        self._drop_unit(p, k)
        self._drop_input(c, k)

    def _drop_unit(self, layer: PackedLayer, k: int) -> None:
        rows = _unpack(layer)
        del rows[k]
        new = pack_layer(layer.n_in, rows, layer.W.device)
        if layer.unit_probe is not None and layer.unit_probe.numel() == layer.n_out:
            keep = [j for j in range(layer.n_out) if j != k]
            new.unit_probe = layer.unit_probe[keep]
            new.unit_born = layer.unit_born[keep]
        self._replace(layer, new)

    def _drop_input(self, layer: PackedLayer, j: int) -> None:
        keep = [i for i in range(layer.n_in) if i != j]
        layer.W = layer.W[:, keep]
        layer.mW = layer.mW[:, keep]
        layer.vW = layer.vW[:, keep]
        layer.n_in -= 1
        layer.ready()

    def _insert_depth_probe(self, g: int, ctx: EditContext) -> None:
        c = self.layers[g]
        alpha, beta = _fold_from_stats(c, u_to_x=True)
        if g > 0 and not self.layers[g - 1].layer_probe and not self.through:
            k = self._probe_unit(self.layers[g - 1])
            if k is not None:
                self._drop_coordinate(g - 1, k)
                keep = [j for j in range(alpha.shape[0]) if j != k]
                alpha, beta = alpha[keep], beta[keep]
        d = c.n_in
        units = []
        eye = torch.eye(d)
        for k in range(d):
            w = _noise((d,), ctx.noise, ctx.generator, c.W).cpu() + eye[k]
            b = float(_noise((), ctx.noise, ctx.generator, c.b))
            units.append([(w, b, 1.0, 0, ctx.step, 0)])
        layer = pack_layer(d, units, c.W.device)
        layer.layer_probe = 1
        layer.layer_born = ctx.step
        layer.track_stats = self._tracking
        _fold_apply(c, alpha, beta)
        c.reset_stats()
        self.layers.insert(g, layer.ready())

    def _remove_identity_layer(self, i: int) -> None:
        l, c = self.layers[i], self.layers[i + 1]
        alpha, beta = _fold_from_stats(l, u_to_x=False)
        _fold_apply(c, alpha, beta)
        c.reset_stats()
        del self.layers[i]

    def ablate(self, item: Item):
        if item.axis == WIDTH:
            i, k = item.address
            c = self._consumer(i)
            snap = c.W[:, k].clone()

            def undo() -> None:
                c.W[:, k] = snap
            c.W[:, k] = 0.0
            return undo
        if item.axis == DEGREE:
            i, k, r = item.address
            layer = self.layers[i]
            rr = int(layer.ptr[k]) + r
            snap = (layer.W[rr].clone(), float(layer.b[rr]), float(layer.a[rr]))
            layer.W[rr] = 0.0
            layer.b[rr] = 1.0
            layer.a[rr] = 1.0

            def undo() -> None:
                layer.W[rr].copy_(snap[0])
                layer.b[rr] = snap[1]
                layer.a[rr] = snap[2]
            return undo
        layer = self.layers[item.address[0]]
        if layer.n_in != layer.n_out:
            return None
        snap = (layer.W.clone(), layer.b.clone(), layer.a.clone())
        p = layer.ptr.tolist()
        eye = torch.eye(layer.n_in, dtype=layer.W.dtype, device=layer.W.device)
        layer.W.zero_()
        layer.b.fill_(1.0)
        layer.a.fill_(1.0)
        for k in range(layer.n_out):
            r = p[k]
            layer.W[r] = eye[k]
            layer.b[r] = 0.0
            layer.a[r] = 1.0

        def undo() -> None:
            layer.W.copy_(snap[0])
            layer.b.copy_(snap[1])
            layer.a.copy_(snap[2])
        return undo

    def cost(self, item: Item) -> int:
        if item.axis == WIDTH:
            i, k = item.address
            p, c = self.layers[i], self._consumer(i)
            n_or = p.num_ors()[k]
            return n_or * (p.n_in + 2) + c.n_or
        if item.axis == DEGREE:
            return self.layers[item.address[0]].n_in + 2
        return self.layers[item.address[0]].num_params()

    def trial_remove(self, item: Item) -> Trial | None:
        i = item.address[0]
        if i + 1 >= len(self.layers):
            return None
        l, c = self.layers[i], self.layers[i + 1]
        if l.n_in != l.n_out:
            return None
        snap = (c.W.clone(), c.b.clone(), c.mW.clone(), c.vW.clone())
        self._remove_identity_layer(i)

        def undo() -> None:
            self.layers.insert(i, l)
            c.W, c.b, c.mW, c.vW = snap

        return Trial(undo=undo, commit=c.reset_stats)

    def removal_order(self, axis: str) -> Sequence[Item]:
        if axis == DEPTH:
            return [Item(DEPTH, (i,)) for i in range(len(self.layers) - 1)]
        return []

    def prune_candidates(self) -> Sequence[Item]:
        out = []
        L = self.layers
        for i, l in enumerate(L):
            p = l.ptr.tolist()
            for k in range(l.n_out):
                for r in range(p[k + 1] - p[k]):
                    out.append(Item(DEGREE, (i, k, r)))
            if i + 1 >= len(L) or l.layer_probe:
                continue
            out += [Item(WIDTH, (i, k)) for k in range(l.n_out)]
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
        if item.axis == DEGREE:
            left = self.layers[i].num_ors()[k] - sum(
                1 for a in ors if a[0] == i and a[1] == k)
            return left > 1
        left = self.layers[i].n_out - sum(1 for a in units if a[0] == i)
        return left > 1

    def params_without(self, removed: Sequence[Item]) -> int:
        units, ors = self._gone(removed)
        total, n_in = 0, self.layers[0].n_in
        for i, l in enumerate(self.layers):
            n_out = 0
            nos = l.num_ors()
            for k, n_or in enumerate(nos):
                if (i, k) in units:
                    continue
                n_out += 1
                gone = sum(1 for a in ors if a[0] == i and a[1] == k)
                total += (n_or - gone) * (n_in + 2)
            n_in = n_out
        return total

    def commit_removals(self, removed: Sequence[Item],
                        axes: Sequence[str] = (WIDTH, DEGREE)) -> dict[str, int]:
        units, ors = self._gone(removed)
        c = {"or_add": 0, "or_drop": 0, "and_add": 0, "and_drop": 0}
        do_width, do_degree = WIDTH in axes, DEGREE in axes
        L = self.layers
        means = {}
        for i in range(len(L) - 1):
            cons = self._consumer(i)
            if cons.stat_ns > 0 and cons.sx is not None:
                means[i] = cons.sx / cons.stat_ns
        for i in reversed(range(len(L))):
            l = L[i]
            drop_w = do_width and i + 1 < len(L) and not l.layer_probe
            for k in reversed(range(l.n_out)):
                gone = (i, k) in units
                if do_degree:
                    n_or = l.num_ors()[k]
                    for r in reversed(range(n_or)):
                        was = int(l.probe[int(l.ptr[k]) + r])
                        if (i, k, r) in ors and not gone:
                            self._drop_or(i, k, r)
                            if not was:
                                c["and_drop"] += 1
                        elif was and not gone:
                            l.probe[int(l.ptr[k]) + r] = 0
                            c["and_add"] += 1
                if drop_w:
                    l.ready()
                    was = int(l.unit_probe[k]) if k < l.unit_probe.numel() else 0
                    if gone:
                        m = means.get(i)
                        self._drop_coordinate(i, k, None if m is None else float(m[k]))
                        if not was:
                            c["or_drop"] += 1
                    elif was:
                        l.unit_probe[k] = 0
                        c["or_add"] += 1
        return c


def _unpack(layer: PackedLayer):
    rows = []
    p = layer.ptr.tolist()
    for k in range(layer.n_out):
        unit = []
        for r in range(p[k], p[k + 1]):
            unit.append((
                layer.W[r].detach().cpu(),
                float(layer.b[r]), float(layer.a[r]),
                int(layer.t[r]), int(layer.born[r]), int(layer.probe[r]),
            ))
        rows.append(unit)
    return rows


def _copy_meta(old: PackedLayer, new: PackedLayer) -> None:
    new.layer_probe = old.layer_probe
    new.layer_born = old.layer_born
