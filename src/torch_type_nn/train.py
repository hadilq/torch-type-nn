"""A training loop that grows and prunes a :class:`TypeNN` while it fits."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .network import TypeNN
from .optim import TypeAdam
from .scaling import Counters, StructureScaler

__all__ = ["fit", "FitResult", "mse_loss"]


def mse_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """0.5 mean((y - t)^2): per sample its gradient is (y - t) / m, as in type-nn."""
    return 0.5 * ((y - t) ** 2).mean()


@dataclass
class FitResult:
    model: TypeNN
    counters: Counters
    history: list[dict] = field(default_factory=list)


def fit(
    model: TypeNN,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int = 1,
    shuffle: bool | Callable[[int, int], torch.Tensor] = True,
    generator: torch.Generator | None = None,
    scale: bool = True,
    rule: str = "threshold",
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
    callback: Callable[[int, StructureScaler], None] | None = None,
) -> FitResult:
    """Train ``model`` on ``(X, Y)`` with :class:`TypeAdam`, growing and pruning its structure.

    Args:
        epochs, lr: the schedule and Adam learning rate.
        batch_size: 1 reproduces type-nn's per-sample protocol; larger
            batches are the fast path (one optimizer step per batch).
        shuffle: ``True`` for a fresh permutation per epoch, ``False`` for
            none, or ``fn(epoch, n) -> LongTensor`` for a custom order.
        scale: ``False`` trains the structure as it is (a static network).
        rule: the scaling rule, ``"threshold"`` (default) or ``"bic"``; see
            :class:`~torch_type_nn.scaling.StructureScaler`.
        callback: called as ``callback(epoch, scaler)`` after each epoch.
    """
    n = X.shape[0]
    steps = math.ceil(n / batch_size)
    opt = TypeAdam(model, lr=lr)
    scaler = StructureScaler(model, opt, epochs=epochs, steps_per_epoch=steps, rule=rule)
    if scale:
        scaler.begin()
    model.train()
    result = FitResult(model, scaler.counters)
    for ep in range(epochs):
        if callable(shuffle):
            order = shuffle(ep, n)
        elif shuffle:
            order = torch.randperm(n, generator=generator)
        else:
            order = torch.arange(n)
        order = order.to(X.device)
        ep_sq = 0.0
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            x, t = X[idx], Y[idx]
            y = model(x)
            loss = loss_fn(y, t)
            ep_sq += float(((y.detach() - t) ** 2).mean(-1).sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if scale:
                scaler.observe(x, y, t)
            else:
                scaler.step += 1
        if scale:
            scaler.epoch_end()
        result.history.append({"epoch": ep, "train_mse": ep_sq / n,
                               "params": model.num_params(), "depth": model.depth})
        if callback is not None:
            callback(ep, scaler)
    if scale:
        scaler.end()
    model.eval()
    return result
