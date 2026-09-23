"""Larger data: does the scaling rule still work when n is in the thousands?

Two tasks the type-nn board does not cover, trained in mini-batches:

* ``digits``: UCI optical digits (1797 x 64 pixels, 10 classes), one-hot MSE;
* ``friedman``: Friedman #1 regression (10 000 x 10 inputs, of which 5 are
  pure noise): ``y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + e``,
  ``e ~ N(0, 1)``; target min-max scaled on the training rows.

Same 70/30 xorshift split and train-only standardisation as ``bench.py``.
Models: type-nn (TypeAdam), the scaled MLP (stock Adam), and two fixed MLPs
(16 hidden units, the board's baseline rule; and 64).

    python benchmarks/scale.py [digits|friedman|all] [--seeds 3] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402

from torch_type_nn import (  # noqa: E402
    ScalableMLP,
    StructureScaler,
    TypeAdam,
    TypeNN,
    TypeNNAdapter,
    mse_loss,
    readout,
)

D = torch.float64
TASKS = {"digits": (60, 0.003), "friedman": (40, 0.003)}   # epochs, Adam lr


def load(task: str):
    if task == "digits":
        rows = [list(map(float, ln.split(",")))
                for ln in gzip.open(bench.DATA / "digits.csv.gz", "rt").read().splitlines()
                if ln.strip()]
        X = torch.tensor([r[:64] for r in rows], dtype=D)
        Y = torch.nn.functional.one_hot(torch.tensor([int(r[64]) for r in rows]), 10).to(D)
        return X, Y, True
    g = torch.Generator().manual_seed(bench.SPLIT_SEED)
    X = torch.rand(10000, 10, dtype=D, generator=g)
    y = (10 * torch.sin(math.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4] + torch.randn(10000, dtype=D, generator=g))
    return X, y.unsqueeze(1), False


def split(task: str):
    X, Y, classify = load(task)
    n = X.shape[0]
    perm = list(range(n))
    bench.shuffle_inplace(perm, bench.SPLIT_SEED)
    ntr = n * 7 // 10
    tr, te = torch.tensor(perm[:ntr]), torch.tensor(perm[ntr:])
    mean, sd = X[tr].mean(0), X[tr].std(0)
    sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
    X = (X - mean) / sd
    if not classify:
        lo, hi = Y[tr].min(0).values, Y[tr].max(0).values
        Y = (Y - lo) / (hi - lo)
    return X[tr], Y[tr], X[te], Y[te], classify


class FixedMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out)).double()

    def forward(self, x):
        return readout(self.net(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def run(task, kind, seed, B):
    epochs, lr = TASKS[task]
    Xtr, Ytr, Xte, Yte, classify = split(task)
    n_in, n_out, n = Xtr.shape[1], Ytr.shape[1], Xtr.shape[0]
    steps = math.ceil(n / B)
    scaler = None
    if kind in ("type-nn", "type-nn-through"):
        model = TypeNN(n_in, n_out, seed=seed, dtype=D)
        opt = TypeAdam(model, lr=lr)
    elif kind == "mlp-scaled":
        model = ScalableMLP(n_in, n_out, seed=seed, dtype=D)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model = FixedMLP(n_in, n_out, int(kind.split("-")[1]), seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    if kind in ("type-nn", "type-nn-through", "mlp-scaled"):
        target = (TypeNNAdapter(model, width_through_depth_probe=True)
                  if kind == "type-nn-through" else model)
        scaler = StructureScaler(target, opt, epochs=epochs, steps_per_epoch=steps)
        scaler.begin()
    g = torch.Generator().manual_seed(seed)
    curve = []
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, B):
            idx = order[s:s + B]
            x, t = Xtr[idx], Ytr[idx]
            y = model(x)
            loss = mse_loss(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if scaler:
                scaler.observe(x, y, t)
        if scaler:
            scaler.epoch_end()
        curve.append(model.num_params())
    if scaler:
        scaler.end()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        y = model(Xte)
        hold_mse = float(((y - Yte) ** 2).mean())
        acc = float((y.argmax(1) == Yte.argmax(1)).double().mean()) if classify else None
        train_mse = float(((model(Xtr) - Ytr) ** 2).mean())
    shape = (model.structure() if hasattr(model, "structure") else None)
    return {"hold_mse": hold_mse, "hold_acc": acc, "train_mse": train_mse,
            "params": model.num_params(), "train_s": train_s, "structure": shape,
            "params_curve": curve[:: max(1, epochs // 10)],
            "counters": vars(scaler.counters) if scaler else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="all", choices=["all", *TASKS])
    ap.add_argument("--models", default="type-nn,mlp-scaled,mlp-16,mlp-64")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    for task in (list(TASKS) if a.task == "all" else [a.task]):
        for kind in a.models.split(","):
            runs = []
            for s in range(a.seeds):
                r = run(task, kind, bench.SPLIT_SEED + 7919 * (s + 1), a.batch_size)
                runs.append(r)
                print(f"  {task} {kind} seed {s}: hold {r['hold_mse']:.5f} acc {r['hold_acc']} "
                      f"params {r['params']} {r['structure']} {r['train_s']:.0f}s",
                      file=sys.stderr, flush=True)
            hm = [r["hold_mse"] for r in runs]
            out = {"task": task, "impl": kind, "seeds": a.seeds, "batch_size": a.batch_size,
                   "epochs": TASKS[task][0], "lr": TASKS[task][1],
                   "hold_mse": statistics.fmean(hm),
                   "hold_mse_sd": statistics.stdev(hm) if len(hm) > 1 else 0.0,
                   "hold_acc": (statistics.fmean(r["hold_acc"] for r in runs)
                                if runs[0]["hold_acc"] is not None else None),
                   "train_mse": statistics.fmean(r["train_mse"] for r in runs),
                   "params": statistics.fmean(r["params"] for r in runs),
                   "train_s": statistics.fmean(r["train_s"] for r in runs),
                   "runs": runs}
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
