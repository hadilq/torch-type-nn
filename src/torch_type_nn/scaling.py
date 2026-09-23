"""The three scaling problems of type-nn: width (Or), degree (And), depth.

A port of ``type_nn_scale.c`` from https://github.com/hadilq/type-nn.

One dummy rule for all three axes. A *probe* is an identity, and back-prop
is free to move it. A probe that back-prop moved out of the identity
(displacement above ``theta(T) = lr * T^(3/4)``) *and* that pays the
Bayesian-information price of its parameters,

    n ln(MSE_without / MSE_with) > k ln n,

is promoted, and a fresh probe takes its place. Late in training every Or
and every coordinate is a candidate for removal; they are ablated in order
of the damage each does alone, and the ablated set keeps growing while the
criterion ``C = n ln MSE + K ln n`` stays at or below its best value seen
while pruning. ``MSE_without`` is measured: the item is reset to its
identity and the epoch's training pairs are re-evaluated.

Schedule on ``u = step / total``: grow on ``u < 1/3`` (only while the
epoch's training MSE exceeds ``Var(t) / N``), fit on the middle third,
prune on ``u >= 2/3``. All edits happen at epoch boundaries and read only
training data seen in the epoch.

Usage::

    model = TypeNN(n, m)
    opt = TypeAdam(model, lr=lr)
    scaler = StructureScaler(model, opt, epochs=E, steps_per_epoch=S)
    scaler.begin()
    for epoch in range(E):
        for x, t in loader:
            y = model(x)
            loss = 0.5 * (y - t).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            scaler.observe(x, y, t)
        scaler.epoch_end()
    scaler.end()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .layer import AndOr, uniform
from .network import TypeNN

__all__ = ["StructureScaler", "Counters", "phase_at", "GROW", "FIT", "PRUNE", "DONE"]

GROW, FIT, PRUNE, DONE = 0, 1, 2, 3


def phase_at(u: float) -> int:
    if u < 1.0 / 3.0:
        return GROW
    if u < 2.0 / 3.0:
        return FIT
    return PRUNE


@dataclass
class Counters:
    """Probes promoted (``*_add``) and live items pruned (``*_drop``) per axis."""
    or_add: int = 0
    or_drop: int = 0
    and_add: int = 0
    and_drop: int = 0
    layer_add: int = 0
    layer_drop: int = 0


@dataclass
class _Epoch:
    x: list = field(default_factory=list)
    t: list = field(default_factory=list)
    n: int = 0
    loss_sum: float = 0.0
    t_sum: torch.Tensor | None = None
    t_sq: torch.Tensor | None = None


class StructureScaler:
    """Grows and prunes a :class:`TypeNN` while it trains.

    Args:
        model: the network.
        optimizer: the optimizer driving ``model`` (normally
            :class:`~torch_type_nn.optim.TypeAdam`). Its learning rate is the
            ``lr`` of the displacement threshold and of the probe noise.
        epochs, steps_per_epoch: the schedule, ``total = epochs * steps_per_epoch``
            optimizer steps.
        eval_batch: chunk size when the epoch's pairs are re-evaluated.
    """

    def __init__(self, model: TypeNN, optimizer: torch.optim.Optimizer, *, epochs: int,
                 steps_per_epoch: int, eval_batch: int = 65536) -> None:
        self.model = model
        self.optimizer = optimizer
        self.total = int(epochs) * int(steps_per_epoch)
        self.step = 0
        self.phase = GROW
        self.crit_best = math.inf
        self.counters = Counters()
        self.epoch_loss = 0.0
        self.epoch_base = 0.0
        self.n_train = 0
        self.eval_batch = int(eval_batch)
        self._ep = _Epoch()
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    # ----------------------------------------------------------------- public

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def gen(self) -> torch.Generator:
        return self.model.generator

    def begin(self) -> None:
        """Start of training: the first probes (exact identities)."""
        self.phase = GROW
        self.crit_best = math.inf
        for layer in self.model.layers:
            layer.track_stats = True
        self._grow_width(0.0)
        self._grow_and(0.0)
        self._reset_epoch()

    @torch.no_grad()
    def observe(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> None:
        """Record one optimizer step: the inputs, the (pre-update) outputs, the targets."""
        x = x.detach().reshape(-1, self.model.in_features)
        y = y.detach().reshape(-1, self.model.out_features)
        t = t.detach().reshape(-1, self.model.out_features).to(y.dtype)
        ep = self._ep
        ep.x.append(x)
        ep.t.append(t)
        ep.n += x.shape[0]
        ep.loss_sum += float(((y - t) ** 2).mean(1).sum())
        ep.t_sum = t.sum(0) if ep.t_sum is None else ep.t_sum + t.sum(0)
        ep.t_sq = (t * t).sum(0) if ep.t_sq is None else ep.t_sq + (t * t).sum(0)
        self._cache = None
        self.step += 1

    def epoch_end(self) -> None:
        """Structural edits for this epoch boundary."""
        u = self.step / self.total if self.total else 1.0
        ph = phase_at(u)
        if self.phase == DONE:
            return
        self._close_epoch_residual()
        if ph == GROW and self.residual_unexplained():
            # depth first: it reads the junction statistics other edits reset
            self._grow_depth(self.measure_mse())
            base = self.measure_mse()
            self._grow_width(base)
            self._grow_and(base)
        elif ph == PRUNE:
            self._prune_step()
        self.phase = ph
        self._reset_epoch()

    def end(self) -> None:
        """End of training: the last prune boundary; no probe survives."""
        if self.phase != PRUNE:
            self._prune_step()
        self.phase = DONE
        for layer in self.model.layers:
            layer.track_stats = False
            layer.compact()

    # -------------------------------------------------------------- measures

    def threshold(self, age: int) -> float:
        """theta(T) = lr T^(3/4): the geometric mean of noise lr sqrt(T) and drift lr T."""
        return self.lr * float(max(age, 1)) ** 0.75

    def residual_unexplained(self) -> bool:
        """Grow only while the epoch's training MSE exceeds Var(t) / N."""
        n = self.n_train if self.n_train else 1
        return self.epoch_loss > self.epoch_base / n

    def _pairs(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._ep.x:
            return None
        if self._cache is None:
            self._cache = (torch.cat(self._ep.x), torch.cat(self._ep.t))
        return self._cache

    @torch.no_grad()
    def measure_mse(self) -> float:
        """Mean per-output MSE of the current network on the epoch's pairs."""
        pairs = self._pairs()
        if pairs is None:
            return self.epoch_loss
        X, T = pairs
        s = 0.0
        for i in range(0, X.shape[0], self.eval_batch):
            y = self.model(X[i:i + self.eval_batch])
            s += float(((y - T[i:i + self.eval_batch]) ** 2).sum())
        return s / (X.shape[0] * T.shape[1])

    def _n_obs(self) -> float:
        n = self._ep.n if self._ep.n else self.n_train
        return float(n) * self.model.out_features

    def bic_ratio(self, mse_with: float, mse_without: float, k: int) -> float:
        """> 1: an item with k parameters pays for itself."""
        n = self._n_obs()
        if k == 0 or n <= 1.0:
            return math.inf
        if not mse_with > 0.0:
            return math.inf if mse_without > 0.0 else 0.0
        if mse_without <= mse_with:
            return 0.0
        return n * math.log(mse_without / mse_with) / (k * math.log(n))

    def criterion(self, mse: float, K: int) -> float:
        """n ln MSE + K ln n; lower is better."""
        n = self._n_obs()
        lm = math.log(mse) if mse > 0.0 else -math.inf
        return n * lm + K * math.log(n if n > 1.0 else 2.0)

    # ------------------------------------------------ distances from identity

    @staticmethod
    def dev_or(layer: AndOr, k: int, r: int) -> float:
        w, b, a = (layer.weight.detach()[k, r], layer.bias.detach()[k, r],
                   layer.assembly.detach()[k, r])
        s = float((w * w).sum() + (b - 1) ** 2 + (a - 1) ** 2)
        return math.sqrt(s / (layer.in_features + 2))

    @staticmethod
    def dev_column(consumer: AndOr, j: int) -> float:
        live = consumer.mask
        c = int(live.sum())
        if not c:
            return 0.0
        w = consumer.weight.detach()[..., j][live]
        return math.sqrt(float((w * w).sum()) / c)

    @staticmethod
    def _sq_one(layer: AndOr) -> torch.Tensor:
        w, b, a = layer.weight.detach(), layer.bias.detach(), layer.assembly.detach()
        return (w * w).sum(-1) + (b - 1) ** 2 + (a - 1) ** 2            # (m, R)

    @staticmethod
    def _sq_carrier(layer: AndOr) -> torch.Tensor:
        # carrier of coordinate k: w = e_k, b = 0, a = 1 (the Or equals x_k)
        m = layer.out_features
        eye = torch.eye(m, layer.in_features, dtype=layer.weight.dtype,
                        device=layer.weight.device).unsqueeze(1)          # (m, 1, n)
        d = layer.weight.detach() - eye
        return (d * d).sum(-1) + layer.bias.detach() ** 2 + (layer.assembly.detach() - 1) ** 2

    @classmethod
    def dev_layer(cls, layer: AndOr) -> float:
        """RMS distance from the identity layer; inf if the layer is not square."""
        if layer.in_features != layer.out_features:
            return math.inf
        live = layer.mask
        one, car = cls._sq_one(layer), cls._sq_carrier(layer)
        all_one = torch.where(live, one, torch.zeros_like(one)).sum(1, keepdim=True)
        v = torch.where(live, all_one - one + car, torch.full_like(one, math.inf))
        s = float(v.min(1).values.sum()) if layer.degree else math.inf
        c = int(live.sum()) * (layer.in_features + 2)
        return math.sqrt(s / c) if c else math.inf

    # ---------------------------------------------------- ablation (exact undo)

    @staticmethod
    def _snap(layer: AndOr, *names: str) -> dict:
        return {n: getattr(layer, n).detach().clone() for n in names}

    @staticmethod
    @torch.no_grad()
    def _restore(layer: AndOr, snap: dict) -> None:
        for n, v in snap.items():
            getattr(layer, n).copy_(v)

    @staticmethod
    @torch.no_grad()
    def _ablate_or(layer: AndOr, k: int, r: int) -> tuple:
        s = (layer.weight[k, r].clone(), layer.bias[k, r].clone(), layer.assembly[k, r].clone())
        layer.weight[k, r] = 0.0
        layer.bias[k, r] = 1.0
        layer.assembly[k, r] = 1.0
        return s

    @staticmethod
    @torch.no_grad()
    def _restore_or(layer: AndOr, k: int, r: int, s: tuple) -> None:
        layer.weight[k, r], layer.bias[k, r], layer.assembly[k, r] = s

    @staticmethod
    @torch.no_grad()
    def _ablate_col(consumer: AndOr, j: int) -> torch.Tensor:
        s = consumer.weight[..., j].clone()
        consumer.weight[..., j] = 0.0
        return s

    @staticmethod
    @torch.no_grad()
    def _restore_col(consumer: AndOr, j: int, s: torch.Tensor) -> None:
        consumer.weight[..., j] = s

    @torch.no_grad()
    def _ablate_layer(self, layer: AndOr) -> dict:
        """Reset a square layer to its identity: per unit the live Or nearest
        e_k becomes the carrier, every other Or the unit identity."""
        snap = self._snap(layer, "weight", "bias", "assembly")
        live = layer.mask
        d = torch.where(live, self._sq_carrier(layer) - self._sq_one(layer),
                        torch.full_like(layer.bias, math.inf))
        best = d.argmin(1)                                               # (m,)
        layer.weight.masked_fill_(live.unsqueeze(-1), 0.0)
        layer.bias.masked_fill_(live, 1.0)
        layer.assembly.fill_(1.0)
        ks = torch.arange(layer.out_features, device=best.device)
        has = live[ks, best]
        ks, best = ks[has], best[has]
        layer.weight[ks, best, ks] = 1.0
        layer.bias[ks, best] = 0.0
        return snap

    def _evidence_or(self, base: float, layer: AndOr, k: int, r: int) -> float:
        s = self._ablate_or(layer, k, r)
        without = self.measure_mse()
        self._restore_or(layer, k, r, s)
        return self.bic_ratio(base, without, layer.in_features + 2)

    def _coordinate_params(self, i: int, j: int) -> int:
        p, c = self.model.layers[i], self.model.layers[i + 1]
        return int(p.mask[j].sum()) * (p.in_features + 2) + int(c.mask.sum())

    def _evidence_coordinate(self, base: float, i: int, j: int) -> float:
        c = self.model.layers[i + 1]
        s = self._ablate_col(c, j)
        without = self.measure_mse()
        self._restore_col(c, j, s)
        return self.bic_ratio(base, without, self._coordinate_params(i, j))

    def _evidence_layer(self, base: float, layer: AndOr) -> float:
        if layer.in_features != layer.out_features:
            return math.inf
        snap = self._ablate_layer(layer)
        without = self.measure_mse()
        self._restore(layer, snap)
        return self.bic_ratio(base, without, layer.num_params())

    # ------------------------------------------------------------------ folds

    @staticmethod
    def _fold_from_stats(at: AndOr, u_to_x: bool) -> tuple[torch.Tensor, torch.Tensor]:
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

    @staticmethod
    @torch.no_grad()
    def _fold_apply(c: AndOr, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        J = min(alpha.shape[0], c.in_features)
        a, b = alpha[:J], beta[:J]
        c.bias += (c.weight[..., :J] * b).sum(-1)
        c.weight[..., :J] *= a
        c.adam_m_weight[..., :J] /= a
        c.adam_v_weight[..., :J] /= a * a

    # ----------------------------------------------------------------- probes

    @staticmethod
    def _probe_or(layer: AndOr, k: int) -> int | None:
        r = (layer.or_probe[k] & layer.mask[k]).nonzero()
        return int(r[0]) if len(r) else None

    @staticmethod
    def _probe_unit(layer: AndOr) -> int | None:
        k = layer.unit_probe.nonzero()
        return int(k[0]) if len(k) else None

    def _probe_layer(self) -> int | None:
        for i, layer in enumerate(self.model.layers):
            if layer.probe:
                return i
        return None

    def _add_and_probe(self, layer: AndOr, k: int) -> None:
        n, lr = layer.in_features, self.lr
        w = uniform((n,), lr, self.gen, layer.weight)
        b = 1.0 + float(uniform((), lr, self.gen, layer.bias))
        layer.add_or(k, w, b, 1.0, probe=True, born=self.step)

    def _add_width_probe(self, i: int) -> None:
        """The producing layer grows a new, trained output unit; the consuming
        layer meets it with weights born at exactly 0."""
        p, c = self.model.layers[i], self.model.layers[i + 1]
        k = p.add_unit()
        bound = 1.0 / math.sqrt(max(p.in_features, 1))
        w = uniform((p.in_features,), bound, self.gen, p.weight)
        b = float(uniform((), bound, self.gen, p.bias))
        p.add_or(k, w, b, 1.0, born=self.step)
        p.unit_probe[k] = True
        p.unit_born[k] = self.step
        c.add_input()

    @torch.no_grad()
    def _drop_coordinate(self, i: int, k: int) -> None:
        """Drop output coordinate k of layer i; the consumer keeps the mean
        of what it contributed (b += w E[x_k])."""
        p, c = self.model.layers[i], self.model.layers[i + 1]
        ns = float(c.stat_ns)
        if ns > 0:
            c.bias += c.weight[..., k] * (c.stat_sx[k] / ns)
        p.drop_unit(k)
        c.drop_input(k)

    @torch.no_grad()
    def _insert_depth_probe(self, g: int) -> None:
        """Identity layer of width d at gap g (in front of layer g)."""
        layers = self.model.layers
        c = layers[g]
        alpha, beta = self._fold_from_stats(c, u_to_x=True)
        # A width probe on the junction in front of c would be a coordinate
        # the identity layer has to carry; retire it first.
        if g > 0 and not layers[g - 1].probe:
            k = self._probe_unit(layers[g - 1])
            if k is not None:
                self._drop_coordinate(g - 1, k)
                keep = [j for j in range(alpha.shape[0]) if j != k]
                alpha, beta = alpha[keep], beta[keep]
        d, lr = c.in_features, self.lr
        l = AndOr(d, d, 1, device=c.weight.device, dtype=c.weight.dtype)
        w = uniform((d, 1, d), lr, self.gen, l.weight)
        w[:, 0, :] += torch.eye(d, dtype=w.dtype, device=w.device)
        l.weight.copy_(w)
        l.bias.copy_(uniform((d, 1), lr, self.gen, l.bias))
        l.assembly.fill_(1.0)
        l.or_born.fill_(self.step)
        l.probe, l.born, l.track_stats = True, self.step, True
        self._fold_apply(c, alpha, beta)
        c.reset_stats()
        self.model.insert_layer(g, l)

    def _remove_identity_layer(self, i: int) -> None:
        """Remove a (near-)identity layer; layer i+1 absorbs it."""
        l, c = self.model.layers[i], self.model.layers[i + 1]
        alpha, beta = self._fold_from_stats(l, u_to_x=False)
        self._fold_apply(c, alpha, beta)
        c.reset_stats()
        self.model.remove_layer(i)

    def _loudest_gap(self, probe: int | None) -> int:
        """Gap with the largest mean |dL/dx| over the epoch, counted in the
        stack without the depth probe. The gap behind the last layer has no
        consumer to absorb a fold and is not a candidate."""
        def loud(layer: AndOr) -> float:
            ns = float(layer.stat_ns)
            return float(layer.stat_gin) / ns if ns else 0.0

        best_g, best, g = 0, -1.0, 0
        for i, layer in enumerate(self.model.layers):
            if i == probe:
                continue
            v = loud(layer)
            if probe is not None and i == probe + 1:
                v = max(v, loud(self.model.layers[probe]))
            if v > best:
                best, best_g = v, g
            g += 1
        return best_g

    # ------------------------------------------------------------------- grow

    def _grow_depth(self, base: float) -> None:
        p = self._probe_layer()
        if p is not None:
            pl = self.model.layers[p]
            if (self.dev_layer(pl) > self.threshold(self.step - pl.born)
                    and self._evidence_layer(base, pl) > 1.0):
                pl.probe = False                   # promoted: a live layer
                self.counters.layer_add += 1
            else:
                g = self._loudest_gap(p)
                if g == p:
                    return                         # already where it is loudest
                self._remove_identity_layer(p)
                self._insert_depth_probe(g)
                return
        self._insert_depth_probe(self._loudest_gap(None))

    def _grow_width(self, base: float) -> None:
        layers = self.model.layers
        for i in range(len(layers) - 1):
            if layers[i].probe or layers[i + 1].probe:
                continue
            pl = layers[i]
            k = self._probe_unit(pl)
            if k is not None:
                age = self.step - int(pl.unit_born[k])
                if (self.dev_column(layers[i + 1], k) > self.threshold(age)
                        and self._evidence_coordinate(base, i, k) > 1.0):
                    pl.unit_probe[k] = False
                    self.counters.or_add += 1
                    k = None
            if k is None:
                self._add_width_probe(i)

    def _grow_and(self, base: float) -> None:
        for layer in self.model.layers:
            for k in range(layer.out_features):
                r = self._probe_or(layer, k)
                if r is not None:
                    age = self.step - int(layer.or_born[k, r])
                    if (self.dev_or(layer, k, r) > self.threshold(age)
                            and self._evidence_or(base, layer, k, r) > 1.0):
                        layer.or_probe[k, r] = False
                        self.counters.and_add += 1
                        r = None
                if r is None:
                    self._add_and_probe(layer, k)

    # ------------------------------------------------------------------ prune

    def _try_remove_layer(self, i: int, ref: float) -> bool:
        """Trial: remove square layer i (the consumer absorbs the fold), measure,
        keep the removal only if the criterion stays within ``ref``."""
        layers = self.model.layers
        l, c = layers[i], layers[i + 1]
        if l.in_features != l.out_features:
            return False
        snap = self._snap(c, "weight", "bias", "adam_m_weight", "adam_v_weight")
        alpha, beta = self._fold_from_stats(l, u_to_x=False)
        self._fold_apply(c, alpha, beta)
        self.model.remove_layer(i)
        m = self.measure_mse()
        if self.criterion(m, self.model.num_params()) <= ref:
            c.reset_stats()
            return True
        self.model.insert_layer(i, l)
        self._restore(c, snap)
        return False

    def _prune_depth(self, ref: float) -> None:
        """The depth probe, then at most one live layer per boundary (never the output layer)."""
        p = self._probe_layer()
        if p is not None:
            if self._try_remove_layer(p, ref):
                return
            self.model.layers[p].probe = False
            self.counters.layer_add += 1
        for i in range(self.model.depth - 1):
            if self._try_remove_layer(i, ref):
                self.counters.layer_drop += 1
                return

    def _prune_pool(self, ref: float) -> None:
        layers = self.model.layers
        base = self.measure_mse()
        cands = []                     # (alone, kind, i, k, r, np)
        for i, l in enumerate(layers):
            for k, r in l.mask.nonzero().tolist():
                s = self._ablate_or(l, k, r)
                without = self.measure_mse()
                self._restore_or(l, k, r, s)
                cands.append((without - base, 0, i, k, r, l.in_features + 2))
            if i + 1 < len(layers):
                for k in range(l.out_features):
                    s = self._ablate_col(layers[i + 1], k)
                    without = self.measure_mse()
                    self._restore_col(layers[i + 1], k, s)
                    cands.append((without - base, 1, i, k, 0, self._coordinate_params(i, k)))
        cands.sort(key=lambda c: c[0])

        or_left = {(i, k): int(n) for i, l in enumerate(layers)
                   for k, n in enumerate(l.num_ors().tolist())}
        coord_left = [l.out_features for l in layers]
        unit_gone: set = set()
        or_gone: set = set()
        def remaining(extra_or=None, extra_unit=None) -> int:
            # Exact parameter count of the model with the ablated set removed:
            # (live Ors) x (live inputs + 2) per surviving unit. (The C
            # reference sums a per-item tally, which counts an Or twice when
            # it is dropped alone and again with its coordinate.)
            total, n_in = 0, layers[0].in_features
            for i, l in enumerate(layers):
                n_out = 0
                for k, n_or in enumerate(l.num_ors().tolist()):
                    if (i, k) in unit_gone or (i, k) == extra_unit:
                        continue
                    n_out += 1
                    gone = sum(1 for r in range(l.degree)
                               if (i, k, r) in or_gone or (i, k, r) == extra_or)
                    total += (n_or - gone) * (n_in + 2)
                n_in = n_out
            return total

        # greedy joint ablation: keep adding the next-cheapest item while the
        # smaller model is still no worse than the best seen while pruning
        for _, kind, i, k, r, _np in cands:
            if (i, k) in unit_gone:
                continue
            if kind == 0 and or_left[(i, k)] <= 1:
                continue
            if kind == 1 and coord_left[i] <= 1:
                continue
            l = layers[i]
            if kind == 0:
                s = self._ablate_or(l, k, r)
                K = remaining(extra_or=(i, k, r))
                if self.criterion(self.measure_mse(), K) > ref:
                    self._restore_or(l, k, r, s)
                    continue
                or_gone.add((i, k, r))
                or_left[(i, k)] -= 1
            else:
                s = self._ablate_col(layers[i + 1], k)
                K = remaining(extra_unit=(i, k))
                if self.criterion(self.measure_mse(), K) > ref:
                    self._restore_col(layers[i + 1], k, s)
                    continue
                unit_gone.add((i, k))
                coord_left[i] -= 1

        # remove what was ablated (it is an identity: exact), then clear the
        # probe flags of what stays
        c = self.counters
        for i in reversed(range(len(layers))):
            l = layers[i]
            for k in reversed(range(l.out_features)):
                gone = (i, k) in unit_gone
                for r in reversed(range(l.degree)):
                    if not bool(l.mask[k, r]):
                        continue
                    was_probe = bool(l.or_probe[k, r])
                    if (i, k, r) in or_gone and not gone:
                        l.drop_or(k, r)
                        if not was_probe:
                            c.and_drop += 1
                    elif was_probe and not gone:
                        l.or_probe[k, r] = False
                        c.and_add += 1
                if i + 1 < len(layers):
                    was_probe = bool(l.unit_probe[k])
                    if gone:
                        self._drop_coordinate(i, k)
                        if not was_probe:
                            c.or_drop += 1
                    elif was_probe:
                        l.unit_probe[k] = False
                        c.or_add += 1
        for l in layers:
            l.compact()

    def _prune_step(self) -> None:
        """One prune boundary. The reference is the best criterion seen while
        pruning, so a smaller network is accepted only if it is at least as
        good a model as the best one so far."""
        cur = self.criterion(self.measure_mse(), self.model.num_params())
        self.crit_best = min(self.crit_best, cur)
        self._prune_depth(self.crit_best)
        self._prune_pool(self.crit_best)
        cur = self.criterion(self.measure_mse(), self.model.num_params())
        self.crit_best = min(self.crit_best, cur)

    # ------------------------------------------------------------- bookkeeping

    def _close_epoch_residual(self) -> None:
        ep = self._ep
        n, m = ep.n, self.model.out_features
        self.epoch_loss = ep.loss_sum / n if n else 0.0
        base = 0.0
        if n and m:
            mu = ep.t_sum / n
            v = ep.t_sq / n - mu * mu
            base = float(v.clamp(min=0).sum()) / m
        self.epoch_base = base
        if n:
            self.n_train = n

    def _reset_epoch(self) -> None:
        for layer in self.model.layers:
            layer.reset_stats()
        self._ep = _Epoch()
        self._cache = None

