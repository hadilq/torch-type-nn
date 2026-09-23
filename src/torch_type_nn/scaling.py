"""The three scaling problems of type-nn: width (Or), degree (And), depth.

A port of ``type_nn_scale.c`` from https://github.com/hadilq/type-nn, written
against the :class:`~torch_type_nn.protocol.Scalable` protocol: this module
holds only the rule (thresholds, evidence, schedule, the greedy prune pool);
what an item *is* comes from an adapter. :class:`TypeNN` models are wrapped in
:class:`~torch_type_nn.adapters.typenn.TypeNNAdapter` automatically.

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

Any optimizer works. With :class:`~torch_type_nn.optim.TypeAdam` every Or has
its own step count and structural edits carry its state exactly. With a stock
``torch.optim`` optimizer the scaler remaps the optimizer state through the
edits (:func:`~torch_type_nn.edits.follow_structure`); new slices then share
their tensor's step count, and depth folds rescale weights but not the
optimizer's moments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from .edits import follow_structure
from .protocol import DEGREE, DEPTH, WIDTH, EditContext, Item, Scalable

__all__ = ["StructureScaler", "Counters", "phase_at", "as_scalable",
           "GROW", "FIT", "PRUNE", "DONE"]

GROW, FIT, PRUNE, DONE = 0, 1, 2, 3


def phase_at(u: float) -> int:
    if u < 1.0 / 3.0:
        return GROW
    if u < 2.0 / 3.0:
        return FIT
    return PRUNE


@dataclass
class Counters:
    """Probes promoted (``*_add``) and live items pruned (``*_drop``) per axis:
    ``or`` = width, ``and`` = degree, ``layer`` = depth."""
    or_add: int = 0
    or_drop: int = 0
    and_add: int = 0
    and_drop: int = 0
    layer_add: int = 0
    layer_drop: int = 0

    _AXIS = {WIDTH: "or", DEGREE: "and", DEPTH: "layer"}

    def add(self, axis: str) -> None:
        name = f"{self._AXIS[axis]}_add"
        setattr(self, name, getattr(self, name) + 1)

    def drop(self, axis: str) -> None:
        name = f"{self._AXIS[axis]}_drop"
        setattr(self, name, getattr(self, name) + 1)

    def update(self, inc: dict[str, int]) -> None:
        for k, v in inc.items():
            setattr(self, k, getattr(self, k) + v)


@dataclass
class _Epoch:
    x: list = field(default_factory=list)
    t: list = field(default_factory=list)
    n: int = 0
    loss_sum: float = 0.0
    t_sum: torch.Tensor | None = None
    t_sq: torch.Tensor | None = None


def as_scalable(model) -> Scalable:
    """A :class:`Scalable` for ``model``: adapters pass through; :class:`TypeNN`
    and :class:`~torch_type_nn.mlp.ScalableMLP` are wrapped."""
    if isinstance(model, Scalable):
        return model
    from .adapters import MLPAdapter, TypeNNAdapter
    from .mlp import ScalableMLP
    from .network import TypeNN
    if isinstance(model, TypeNN):
        return TypeNNAdapter(model)
    if isinstance(model, ScalableMLP):
        return MLPAdapter(model)
    raise TypeError(f"no scaling adapter for {type(model).__name__}; "
                    "implement torch_type_nn.protocol.Scalable")


class StructureScaler:
    """Grows and prunes a model while it trains, by type-nn's rule.

    Args:
        model: a :class:`TypeNN`, or any :class:`~torch_type_nn.protocol.Scalable` adapter.
        optimizer: the optimizer driving the model. Its learning rate is the
            ``lr`` of the displacement threshold and of the probe noise.
        epochs, steps_per_epoch: the schedule, ``total = epochs * steps_per_epoch``
            optimizer steps.
        eval_batch: chunk size when the epoch's pairs are re-evaluated.
        generator: RNG for probe initialisation; default ``model.generator``.
    """

    def __init__(self, model, optimizer: torch.optim.Optimizer, *, epochs: int,
                 steps_per_epoch: int, eval_batch: int = 65536,
                 generator: torch.Generator | None = None) -> None:
        self.target: Scalable = as_scalable(model)
        self.model: nn.Module = self.target.model
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
        self._gen = generator
        self._n_out = getattr(self.model, "out_features", None)
        self._ep = _Epoch()
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    # ----------------------------------------------------------------- public

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def gen(self) -> torch.Generator:
        return self._gen if self._gen is not None else self.model.generator

    def _ctx(self) -> EditContext:
        return EditContext(step=self.step, noise=self.lr, generator=self.gen)

    def begin(self) -> None:
        """Start of training: the first probes (exact identities)."""
        self.phase = GROW
        self.crit_best = math.inf
        self.target.set_tracking(True)
        with self._editing():
            self._grow_axis(WIDTH, 0.0)
            self._grow_axis(DEGREE, 0.0)
        self._reset_epoch()

    @torch.no_grad()
    def observe(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> None:
        """Record one optimizer step: the inputs, the (pre-update) outputs, the targets."""
        m = y.shape[-1]
        self._n_out = m
        y = y.detach().reshape(-1, m)
        t = t.detach().reshape(-1, m).to(y.dtype)
        ep = self._ep
        ep.x.append(x.detach())
        ep.t.append(t)
        ep.n += t.shape[0]
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
        with self._editing():
            if ph == GROW and self.residual_unexplained():
                # depth first: it reads the junction statistics other edits reset
                self._grow_depth(self.measure_mse())
                base = self.measure_mse()
                self._grow_axis(WIDTH, base)
                self._grow_axis(DEGREE, base)
            elif ph == PRUNE:
                self._prune_step()
        self.phase = ph
        self._reset_epoch()

    def end(self) -> None:
        """End of training: the last prune boundary; no probe survives."""
        with self._editing():
            if self.phase != PRUNE:
                self._prune_step()
            self.phase = DONE
            self.target.set_tracking(False)
            self.target.finalize()

    # ------------------------------------------------------- optimizer sync

    def _editing(self):
        return _EditScope(self)

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
        """Mean per-output MSE of the current model on the epoch's pairs."""
        pairs = self._pairs()
        if pairs is None:
            return self.epoch_loss
        X, T = pairs
        m = T.shape[1]
        s, done = 0.0, 0
        for i in range(0, X.shape[0], self.eval_batch):
            y = self.model(X[i:i + self.eval_batch]).reshape(-1, m)
            s += float(((y - T[done:done + y.shape[0]]) ** 2).sum())
            done += y.shape[0]
        return s / (T.shape[0] * m)

    def _n_obs(self) -> float:
        n = self._ep.n if self._ep.n else self.n_train
        return float(n) * (self._n_out or 1)

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

    def evidence(self, base: float, item: Item) -> float:
        """BIC ratio of ``item``: ablate it, re-measure, undo exactly."""
        undo = self.target.ablate(item)
        if undo is None:
            return math.inf
        without = self.measure_mse()
        undo()
        return self.bic_ratio(base, without, self.target.cost(item))

    def _passes(self, base: float, item: Item) -> bool:
        """Moved out of the identity, and pays for its parameters."""
        age = self.step - self.target.born(item)
        return (self.target.displacement(item) > self.threshold(age)
                and self.evidence(base, item) > 1.0)

    # ------------------------------------------------------------------- grow

    def _grow_axis(self, axis: str, base: float) -> None:
        tgt = self.target
        for site in tgt.probe_sites(axis):
            item = tgt.probe_at(axis, site)
            if item is not None and self._passes(base, item):
                tgt.promote(item)
                self.counters.add(axis)
                item = None
            if item is None:
                tgt.add_probe(axis, site, self._ctx())

    def _grow_depth(self, base: float) -> None:
        tgt = self.target
        p = tgt.probe_at(DEPTH)
        if p is not None:
            if self._passes(base, p):
                tgt.promote(p)
                self.counters.add(DEPTH)
            else:
                g = tgt.best_site(DEPTH, moving=p)
                if g == tgt.site_of(p):
                    return                         # already where it is loudest
                tgt.remove_probe(p)
                tgt.add_probe(DEPTH, g, self._ctx())
                return
        tgt.add_probe(DEPTH, tgt.best_site(DEPTH), self._ctx())

    # ------------------------------------------------------------------ prune

    def _try_remove(self, item: Item, ref: float) -> bool:
        """Remove ``item`` tentatively; keep it only if the criterion stays within ``ref``."""
        trial = self.target.trial_remove(item)
        if trial is None:
            return False
        if self.criterion(self.measure_mse(), self.target.num_params()) <= ref:
            trial.commit()
            return True
        trial.undo()
        return False

    def _prune_depth(self, ref: float) -> None:
        """The depth probe, then at most one live layer per boundary."""
        tgt = self.target
        p = tgt.probe_at(DEPTH)
        if p is not None:
            if self._try_remove(p, ref):
                return
            tgt.promote(p)
            self.counters.add(DEPTH)
        for item in tgt.removal_order(DEPTH):
            if self._try_remove(item, ref):
                self.counters.drop(DEPTH)
                return

    def _prune_pool(self, ref: float) -> None:
        tgt = self.target
        base = self.measure_mse()
        cands = []
        for item in tgt.prune_candidates():
            undo = tgt.ablate(item)
            without = self.measure_mse()
            undo()
            cands.append((without - base, item))
        cands.sort(key=lambda c: c[0])            # stable: ties keep candidate order
        # greedy joint ablation: keep adding the next-cheapest item while the
        # smaller model is still no worse than the best seen while pruning
        removed: list[Item] = []
        for _, item in cands:
            if not tgt.can_remove(item, removed):
                continue
            undo = tgt.ablate(item)
            K = tgt.params_without([*removed, item])
            if self.criterion(self.measure_mse(), K) > ref:
                undo()
                continue
            removed.append(item)
        self.counters.update(tgt.commit_removals(removed))

    def _prune_step(self) -> None:
        """One prune boundary. The reference is the best criterion seen while
        pruning, so a smaller model is accepted only if it is at least as good
        a model as the best one so far."""
        cur = self.criterion(self.measure_mse(), self.target.num_params())
        self.crit_best = min(self.crit_best, cur)
        self._prune_depth(self.crit_best)
        self._prune_pool(self.crit_best)
        cur = self.criterion(self.measure_mse(), self.target.num_params())
        self.crit_best = min(self.crit_best, cur)

    # ------------------------------------------------------------- bookkeeping

    def _close_epoch_residual(self) -> None:
        ep = self._ep
        n, m = ep.n, (self._n_out or 0)
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
        self.target.reset_stats()
        self._ep = _Epoch()
        self._cache = None


class _EditScope:
    """Collect parameter replacements during structural edits and, for a
    stock optimizer, remap its state when the edits are done."""

    def __init__(self, scaler: StructureScaler) -> None:
        self.s = scaler
        self.edits: list = []

    def __enter__(self):
        self.s.target.set_edit_sink(self.edits)
        return self

    def __exit__(self, *exc) -> None:
        # TypeAdam keeps AndOr state inside the layers (the remap finds none
        # to move) but holds ordinary Adam state for any other parameter.
        self.s.target.set_edit_sink(None)
        follow_structure(self.s.optimizer, self.s.model, self.edits)
