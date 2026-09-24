"""Larger data: does the scaling rule still work when n is in the thousands?

Two tasks the type-nn board does not cover, trained in mini-batches:

* ``digits``: UCI optical digits (1797 x 64 pixels, 10 classes), one-hot MSE;
* ``friedman``: Friedman #1 regression, the fixed Delve / OpenML 564 "fried"
  file (40 768 x 10 inputs, of which 5 are pure noise):
  ``y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + e``, ``e ~ N(0, 1)``;
  target min-max scaled on the training rows. Since ``Var(e) = 1`` is known,
  the best reachable hold-out MSE is ``1 / span^2`` (span: the training range
  of y); it is computed from the data and reported as ``noise_floor``.

Both files are pinned in ``datasets.json`` (fetched by ``nix develop`` or
``fetch_data.py``). Same 70/30 xorshift split and train-only standardisation
as ``bench.py``; the type-nn and scaled-MLP models come from ``bench.MODELS``
(built by ``bench.build`` at the task's lr), plus two fixed MLPs (16 hidden
units, the board's baseline rule; and 64).

    python benchmarks/scale.py [digits|friedman|all] [--models type-nn,mlp-16]
                               [--seeds 5] [--batch-size 32] [--device cuda]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402

from torch_type_nn import (  # noqa: E402
    mse_loss,
    readout,
)

D = torch.float64
TASKS = {"digits": (60, 0.003), "friedman": (40, 0.003)}   # epochs, Adam lr
FILES = {"digits": "digits.csv.gz", "friedman": "564_fried.tsv.gz"}
FIXED = {"mlp-16": 16, "mlp-64": 64}


def load_friedman(path) -> tuple[torch.Tensor, torch.Tensor]:
    """The fried TSV: a header, 10 inputs and the target (``target`` column,
    else the last one)."""
    lines = [ln for ln in gzip.open(path, "rt").read().splitlines() if ln.strip()]
    head = lines[0].split("\t")
    try:
        [float(v) for v in head]
        rows, cols = lines, [f"x{i}" for i in range(len(head))]
    except ValueError:
        rows, cols = lines[1:], [c.strip() for c in head]
    low = [c.lower() for c in cols]
    ti = low.index("target") if "target" in low else len(cols) - 1
    data = torch.tensor([list(map(float, ln.split("\t"))) for ln in rows], dtype=D)
    keep = [j for j in range(len(cols)) if j != ti]
    return data[:, keep], data[:, ti:ti + 1]


def load(task: str):
    path = bench.data_file(FILES[task])
    if task == "digits":
        rows = [list(map(float, ln.split(",")))
                for ln in gzip.open(path, "rt").read().splitlines() if ln.strip()]
        X = torch.tensor([r[:64] for r in rows], dtype=D)
        Y = torch.nn.functional.one_hot(torch.tensor([int(r[64]) for r in rows]), 10).to(D)
        return X, Y, True
    X, Y = load_friedman(path)
    if X.shape[1] != 10:
        raise ValueError(f"{path}: expected 10 inputs, found {X.shape[1]}")
    return X, Y, False


def split(task: str):
    """(Xtr, Ytr, Xte, Yte, classify, noise_floor)."""
    X, Y, classify = load(task)
    n = X.shape[0]
    perm = list(range(n))
    bench.shuffle_inplace(perm, bench.SPLIT_SEED)
    ntr = n * 7 // 10
    tr, te = torch.tensor(perm[:ntr]), torch.tensor(perm[ntr:])
    mean, sd = X[tr].mean(0), X[tr].std(0)
    sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
    X = (X - mean) / sd
    floor = None
    if not classify:
        lo, hi = Y[tr].min(0).values, Y[tr].max(0).values
        Y = (Y - lo) / (hi - lo)
        floor = float((1.0 / (hi - lo) ** 2).mean())      # Var(e) = 1 before scaling
    return X[tr], Y[tr], X[te], Y[te], classify, floor


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


def run(task, kind, seed, B, device="cpu", data=None):
    epochs, lr = TASKS[task]
    Xtr, Ytr, Xte, Yte, classify, _ = data or split(task)
    Xtr, Ytr, Xte, Yte = (t.to(device) for t in (Xtr, Ytr, Xte, Yte))
    n_in, n_out, n = Xtr.shape[1], Ytr.shape[1], Xtr.shape[0]
    steps = math.ceil(n / B)
    if kind in FIXED:
        model = FixedMLP(n_in, n_out, FIXED[kind], seed).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = None
    else:
        model, opt, scaler = bench.build(kind, n_in, n_out, seed, lr, steps, epochs,
                                         device, lr_scale=1.0)
    g = torch.Generator().manual_seed(seed)
    curve = []
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=g).to(device)
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
            "layers": getattr(model, "depth", 2),
            "params_curve": curve[:: max(1, epochs // 10)],
            "counters": dict(vars(scaler.counters)) if scaler else None}


_SPLITS: dict = {}


def _job(args):
    task, kind, s, batch, device = args
    torch.set_num_threads(1)
    if task not in _SPLITS:
        _SPLITS[task] = split(task)
    return run(task, kind, bench.SPLIT_SEED + 7919 * (s + 1), batch, device, _SPLITS[task])


def row(task, kind, runs, a, data):
    hm = [r["hold_mse"] for r in runs]
    layers = [r.get("layers") for r in runs if r.get("layers") is not None]
    return {"task": task, "impl": kind, "seeds": len(runs), "batch_size": a.batch_size,
            "device": a.device, "n_train": data[0].shape[0], "noise_floor": data[5],
            "epochs": TASKS[task][0], "lr": TASKS[task][1],
            "hold_mse": statistics.fmean(hm),
            "hold_mse_sd": statistics.stdev(hm) if len(hm) > 1 else 0.0,
            "hold_acc": (statistics.fmean(r["hold_acc"] for r in runs)
                         if runs[0]["hold_acc"] is not None else None),
            "train_mse": statistics.fmean(r["train_mse"] for r in runs),
            "params": statistics.fmean(r["params"] for r in runs),
            "layers": statistics.fmean(layers) if layers else None,
            "train_s": statistics.fmean(r["train_s"] for r in runs),
            "runs": runs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="all", choices=["all", *TASKS])
    ap.add_argument("--models", default="type-nn,type-nn-bic,mlp-scaled,mlp-16,mlp-64",
                    help="comma-separated: " + ", ".join([*bench.MODELS, *FIXED]))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes, one (task, model, seed) each; 0 = all cores")
    a = ap.parse_args(argv)
    kinds = a.models.split(",")
    unknown = [k for k in kinds if k not in bench.MODELS and k not in FIXED]
    if unknown:
        ap.error(f"unknown model(s) {unknown}")
    if a.device == "cpu":
        torch.set_num_threads(1)
    tasks = list(TASKS) if a.task == "all" else [a.task]
    jobs = a.jobs or os.cpu_count() or 1
    todo = [(t, k, s, a.batch_size, a.device) for t in tasks for k in kinds
            for s in range(a.seeds)]
    if jobs > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_job, todo))
    else:
        results = []
        for j in todo:
            results.append(_job(j))
            r = results[-1]
            print(f"  {j[0]} {j[1]} seed {j[2]}: hold {r['hold_mse']:.5f} acc {r['hold_acc']} "
                  f"params {r['params']} {r['structure']} {r['train_s']:.0f}s",
                  file=sys.stderr, flush=True)
    i = 0
    for task in tasks:
        data = _SPLITS.get(task) or split(task)
        for kind in kinds:
            print(json.dumps(row(task, kind, results[i:i + a.seeds], a, data)), flush=True)
            i += a.seeds


if __name__ == "__main__":
    main()
