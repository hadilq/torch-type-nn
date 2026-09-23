"""The type-nn board, reproduced with torch-type-nn.

A line-by-line port of ``bench.c`` from https://github.com/hadilq/type-nn:
the same data files and parsing, the same fixed 70/30 split (xorshift32
Fisher-Yates, seed 34972), train-only standardisation, the same per-epoch
shuffle, per-sample Adam at ``lr * 0.1``, epochs and lr per task, and 5
initialisation seeds per cell. What differs is the initialisation RNG
(torch's generator instead of xorshift32), so single runs are not
bit-identical to C; the board compares means over seeds.

    python benchmarks/bench.py [task|all] [type-nn|mlp|all] [--seeds 5] [--batch-size 1]

Prints one JSON line per (task, model), like ``./bench``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

from torch_type_nn import StructureScaler, TypeAdam, TypeNN, mse_loss, readout

SPLIT_SEED = 34972
LR_SCALE = 0.1
MASK = 0xFFFFFFFF
DATA = Path(__file__).parent / "data"
TASKS = {  # name: (file, epochs, lr)
    "xor": (None, 2000, 0.08),
    "iris": ("iris.data", 250, 0.05),
    "wine": ("wine.data", 200, 0.03),
    "wdbc": ("wdbc.data", 80, 0.02),
    "diabetes": ("diabetes.tab.txt", 150, 0.02),
    "ionosphere": ("ionosphere.data", 120, 0.02),
}


# ------------------------------------------------------------------ data

def xorshift32(s: int) -> int:
    x = s if s else 2463534242
    x ^= (x << 13) & MASK
    x ^= x >> 17
    x ^= (x << 5) & MASK
    return x


def shuffle_inplace(order: list, seed: int) -> None:
    s = seed
    for i in range(len(order), 1, -1):
        s = xorshift32(s)
        j = s % i
        order[i - 1], order[j] = order[j], order[i - 1]


def one_hot(k: int, n: int) -> list[float]:
    return [1.0 if i == k else 0.0 for i in range(n)]


def load(task: str):
    """Rows (X, Y) and whether the task is classification; parsing as dataset.c."""
    path = DATA / TASKS[task][0]
    X, Y = [], []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if task == "iris":
        names = {"setosa": 0, "versicolor": 1, "virginica": 2}
        for ln in lines:
            p = ln.split(",")
            if len(p) < 5:
                continue
            X.append([float(v) for v in p[:4]])
            Y.append(one_hot(next((v for k, v in names.items() if k in p[4]), 0), 3))
        return X, Y, True
    if task == "wine":
        for ln in lines:
            p = ln.split(",")
            X.append([float(v) for v in p[1:14]])
            Y.append(one_hot(min(max(int(p[0]) - 1, 0), 2), 3))
        return X, Y, True
    if task == "wdbc":
        for ln in lines:
            p = ln.split(",")
            X.append([float(v) for v in p[2:32]])
            Y.append([1.0 if p[1] in ("M", "m") else 0.0])
        return X, Y, True
    if task == "diabetes":
        for ln in lines[1:]:
            v = [float(t) for t in ln.split()]
            X.append(v[:10])
            Y.append([v[10]])
        return X, Y, False
    if task == "ionosphere":
        for ln in lines:
            p = ln.split(",")
            X.append([float(v) for v in p[:34]])
            Y.append([1.0 if p[34].strip()[:1] in ("g", "G") else 0.0])
        return X, Y, True
    raise ValueError(task)


def split(task: str):
    X, Y, classify = load(task)
    n = len(X)
    perm = list(range(n))
    shuffle_inplace(perm, SPLIT_SEED)
    ntr = n * 7 // 10
    Xt = torch.tensor(X, dtype=torch.float64)
    Yt = torch.tensor(Y, dtype=torch.float64)
    tr, te = perm[:ntr], perm[ntr:]
    mean = Xt[tr].mean(0)
    sd = Xt[tr].std(0, unbiased=True)
    sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
    Xt = (Xt - mean) / sd
    if not classify:
        lo, hi = Yt[tr].min(0).values, Yt[tr].max(0).values
        span = torch.where(hi - lo < 1e-12, torch.ones_like(hi), hi - lo)
        Yt = (Yt - lo) / span
    return Xt[tr], Yt[tr], Xt[te], Yt[te], classify


# ---------------------------------------------------------------- models

class MLP(nn.Module):
    """The c-mlp baseline: Linear-ReLU-Linear with the same readout F."""

    def __init__(self, n_in: int, n_out: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        h = max(8 if n_in <= 4 else 16, n_out)
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, n_out)).double()

    def forward(self, x):
        return readout(self.net(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train(kind, n_in, n_out, seed, Xtr, Ytr, epochs, lr, batch, device):
    n = Xtr.shape[0]
    steps = math.ceil(n / batch)
    if kind == "type-nn":
        model = TypeNN(n_in, n_out, seed=seed, dtype=torch.float64).to(device)
        opt = TypeAdam(model, lr=lr * LR_SCALE)
        scaler = StructureScaler(model, opt, epochs=epochs, steps_per_epoch=steps)
        scaler.begin()
    else:
        model = MLP(n_in, n_out, seed).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr * LR_SCALE)
        scaler = None
    model.train()
    for ep in range(epochs):
        order = list(range(n))
        shuffle_inplace(order, SPLIT_SEED ^ (((ep + 1) * 0x9E3779B9) & MASK))
        order_t = torch.tensor(order, device=device)
        for s in range(0, n, batch):
            idx = order_t[s:s + batch]
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
    if scaler:
        scaler.end()
    model.eval()
    return model, scaler


@torch.no_grad()
def metrics(model, X, Y, classify):
    y = model(X)
    mse = float(((y - Y) ** 2).mean())
    acc = None
    if classify:
        if Y.shape[1] == 1:
            acc = float(((y >= 0.5) == (Y >= 0.5)).double().mean())
        else:
            acc = float((y.argmax(1) == Y.argmax(1)).double().mean())
    return mse, acc


@torch.no_grad()
def us_per_infer(model, X, min_s=0.2):
    done, t0 = 0, time.perf_counter()
    while True:
        for r in range(200):
            model(X[r % X.shape[0]:r % X.shape[0] + 1])
        done += 200
        dt = time.perf_counter() - t0
        if dt >= min_s:
            return dt * 1e6 / done


# ------------------------------------------------------------------- board

def cell(task, kind, seeds, batch, device, verbose):
    file, epochs, lr = TASKS[task]
    if file is None:
        Xtr = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float64)
        Ytr = torch.tensor([[0], [1], [1], [0]], dtype=torch.float64)
        Xte = Yte = None
        classify = True
    else:
        Xtr, Ytr, Xte, Yte, classify = split(task)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    if Xte is not None:
        Xte, Yte = Xte.to(device), Yte.to(device)
    n_in, n_out = Xtr.shape[1], Ytr.shape[1]
    rec = {k: [] for k in ("hold_mse", "hold_acc", "mse", "acc", "params", "train_s",
                           "layers", "or_add", "or_drop", "and_add", "and_drop",
                           "layer_add", "layer_drop")}
    us = None
    for s in range(seeds):
        seed = SPLIT_SEED + 7919 * (s + 1)
        t0 = time.perf_counter()
        model, scaler = train(kind, n_in, n_out, seed, Xtr, Ytr, epochs, lr, batch, device)
        rec["train_s"].append(time.perf_counter() - t0)
        mse, acc = metrics(model, Xtr, Ytr, classify)
        rec["mse"].append(mse)
        rec["acc"].append(acc)
        if Xte is not None:
            hm, ha = metrics(model, Xte, Yte, classify)
            rec["hold_mse"].append(hm)
            rec["hold_acc"].append(ha)
        rec["params"].append(model.num_params())
        if scaler:
            rec["layers"].append(model.depth)
            for k, v in vars(scaler.counters).items():
                rec[k].append(v)
        else:
            rec["layers"].append(2)
        if verbose:
            hold = rec["hold_mse"][-1] if Xte is not None else "n/a"
            print(f"  {task} {kind} seed {s}: hold_mse {hold}"
                  f" train_mse {mse:.2e} params {model.num_params()} layers {rec['layers'][-1]}"
                  f" {rec['train_s'][-1]:.1f}s", file=sys.stderr)
        if s + 1 == seeds:
            us = us_per_infer(model, Xtr)

    def mean(v):
        v = [x for x in v if x is not None]
        return statistics.fmean(v) if v else None

    def sd(v):
        v = [x for x in v if x is not None]
        return statistics.stdev(v) if len(v) > 1 else 0.0

    out = {"impl": "torch-type-nn" if kind == "type-nn" else "torch-mlp", "task": task,
           "seeds": seeds, "epochs": epochs, "lr": lr, "batch_size": batch,
           "hold_mse": mean(rec["hold_mse"]), "hold_mse_sd": sd(rec["hold_mse"]),
           "hold_acc": mean(rec["hold_acc"]), "hold_acc_sd": sd(rec["hold_acc"]),
           "mse": mean(rec["mse"]), "acc": mean(rec["acc"]),
           "params": mean(rec["params"]), "params_sd": sd(rec["params"]),
           "train_s": mean(rec["train_s"]), "us_per_infer": us,
           "init_layers": TypeNN(n_in, n_out).birth_depth if kind == "type-nn" else 2,
           "layers": mean(rec["layers"])}
    for k in ("or_add", "or_drop", "and_add", "and_drop", "layer_add", "layer_drop"):
        out[k] = mean(rec[k]) if rec[k] else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", nargs="?", default="all", choices=["all", *TASKS])
    ap.add_argument("model", nargs="?", default="all", choices=["all", "type-nn", "mlp"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    tasks = list(TASKS) if a.task == "all" else [a.task]
    kinds = ["type-nn", "mlp"] if a.model == "all" else [a.model]
    for task in tasks:
        for kind in kinds:
            print(json.dumps(cell(task, kind, a.seeds, a.batch_size, a.device, a.verbose)),
                  flush=True)


if __name__ == "__main__":
    main()
