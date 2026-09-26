"""The type-nn board, reproduced with torch-type-nn.

A line-by-line port of ``bench.c`` from https://github.com/hadilq/type-nn:
the same data files and parsing, the same fixed 70/30 split (xorshift32
Fisher-Yates, seed 34972), train-only standardisation, the same per-epoch
shuffle, per-sample Adam at ``lr * 0.1``, epochs and lr per task, and 5
initialisation seeds per cell. What differs is the initialisation RNG
(torch's generator instead of xorshift32), so single runs are not
bit-identical to C; the board compares means over seeds.

Models (``MODELS``; every one goes through the same ``train`` loop):

    type-nn              the architecture: threshold rule, width on every junction
    type-nn-bic          the same with the BIC prior (rule="bic")
    ref-type-nn          C's type-nn exactly: rule="bic", width_through_depth_probe=False
    ref-type-nn-overfit  C's type-nn-overfit exactly: threshold rule, no width through the probe
    c-type-nn            NativeTypeNN wrapping C type-nn (BIC); board row "C type-nn"
    c-type-nn-overfit    NativeTypeNN wrapping C type-nn-overfit; board row "C type-nn-overfit"
    mlp                  the c-mlp baseline: Linear-ReLU-Linear, 8 or 16 hidden units
    mlp-scaled           ScalableMLP grown and pruned by the default (threshold) rule

    python benchmarks/bench.py [task|all] [model|all] [--seeds 5] [--batch-size 1] [--device cuda]

Data: ``$TORCH_TYPE_NN_DATA`` (set by ``nix develop`` / ``nix run .#bench``),
else ``benchmarks/data`` (filled by ``nix develop`` or
``python benchmarks/fetch_data.py``). Prints one JSON line per (task, model).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

from torch_type_nn import (
    NativeTypeNN,
    ScalableMLP,
    StructureScaler,
    TypeAdam,
    TypeNN,
    TypeNNAdapter,
    mse_loss,
    readout,
)

SPLIT_SEED = 34972
LR_SCALE = 0.1
MASK = 0xFFFFFFFF
DATA = Path(os.environ.get("TORCH_TYPE_NN_DATA") or Path(__file__).parent / "data")
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


def data_file(name: str) -> Path:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Enter `nix develop` (it copies the pinned datasets to "
            "benchmarks/data), run `python benchmarks/fetch_data.py`, or set "
            "TORCH_TYPE_NN_DATA to a directory holding them.")
    return path


def load(task: str):
    """Rows (X, Y) and whether the task is classification; parsing as dataset.c."""
    path = data_file(TASKS[task][0])
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
    # dataset.c: mean / sample-sd over training rows in perm order (not torch.std).
    mean = torch.zeros(Xt.shape[1], dtype=torch.float64)
    for i in tr:
        mean += Xt[i]
    mean = mean / ntr
    var = torch.zeros(Xt.shape[1], dtype=torch.float64)
    for i in tr:
        d = Xt[i] - mean
        var = var + d * d
    sd = torch.sqrt(var / (ntr - 1 if ntr > 1 else 1))
    sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
    Xt = (Xt - mean) / sd
    if not classify:
        lo = Yt[tr[0]].clone()
        hi = lo.clone()
        for i in tr[1:]:
            lo = torch.minimum(lo, Yt[i])
            hi = torch.maximum(hi, Yt[i])
        span = hi - lo
        span = torch.where(span < 1e-12, torch.ones_like(span), span)
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


# kind: (family, scaling rule, width through the depth probe)
# family "c" is NativeTypeNN wrapping C type-nn / type-nn-overfit (ragged Ors).
MODELS = {
    "type-nn": ("type-nn", "threshold", True),
    "type-nn-bic": ("type-nn", "bic", True),
    "ref-type-nn": ("type-nn", "bic", False),
    "ref-type-nn-overfit": ("type-nn", "threshold", False),
    "c-type-nn": ("c", "bic", None),
    "c-type-nn-overfit": ("c", "threshold", None),
    "mlp": ("mlp", None, None),
    "mlp-scaled": ("mlp-scaled", "threshold", True),
}

# JSON impl id written by aggregate (board.py keys).
IMPL = {
    "c-type-nn": "type-nn",
    "c-type-nn-overfit": "type-nn-overfit",
}


def build(kind, n_in, n_out, seed, lr, steps, epochs, device, lr_scale=LR_SCALE):
    """Model, optimizer and (for scaled models) the started scaler. The same
    constructor serves the board (``lr * 0.1``, as bench.c) and scale.py."""
    family, rule, through = MODELS[kind]
    if family == "c":
        raise TypeError("C models are NativeTypeNN; use train_c / train(), not build()")
    lr = lr * lr_scale
    if family == "type-nn":
        model = TypeNN(n_in, n_out, seed=seed, dtype=torch.float64).to(device)
        opt = TypeAdam(model, lr=lr)
        target = TypeNNAdapter(model, width_through_depth_probe=through)
    elif family == "mlp-scaled":
        # born with one hidden layer of max(2, m) units; stock Adam
        model = ScalableMLP(n_in, n_out, seed=seed, dtype=torch.float64).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        target = model
    else:
        model = MLP(n_in, n_out, seed).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        return model, opt, None
    scaler = StructureScaler(target, opt, epochs=epochs, steps_per_epoch=steps, rule=rule)
    scaler.begin()
    return model, opt, scaler


def train_c(kind, n_in, n_out, seed, Xtr, Ytr, epochs, lr):
    """C protocol via NativeTypeNN: one tnn_py_epoch per epoch (bench.c)."""
    family, rule, _ = MODELS[kind]
    assert family == "c"
    net = NativeTypeNN(n_in, n_out, rule=rule, seed=seed)
    Xtr = Xtr.detach().cpu().to(torch.float64).contiguous()
    Ytr = Ytr.detach().cpu().to(torch.float64).contiguous()
    net.begin(Xtr.shape[0], epochs, lr)
    for ep in range(epochs):
        shuf = SPLIT_SEED ^ (((ep + 1) * 0x9E3779B9) & MASK)
        net.epoch(Xtr, Ytr, shuf)
    net.end()
    net.eval()
    return net


def train(kind, n_in, n_out, seed, Xtr, Ytr, epochs, lr, batch, device):
    if MODELS[kind][0] == "c":
        return train_c(kind, n_in, n_out, seed, Xtr, Ytr, epochs, lr), None
    n = Xtr.shape[0]
    steps = math.ceil(n / batch)
    model, opt, scaler = build(kind, n_in, n_out, seed, lr, steps, epochs, device)
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

_DATA_CACHE: dict = {}


def task_data(task):
    """(Xtr, Ytr, Xte, Yte, classify) on the CPU; cached per process."""
    if task not in _DATA_CACHE:
        file, _, _ = TASKS[task]
        if file is None:
            Xtr = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float64)
            Ytr = torch.tensor([[0], [1], [1], [0]], dtype=torch.float64)
            _DATA_CACHE[task] = (Xtr, Ytr, None, None, True)
        else:
            _DATA_CACHE[task] = split(task)
    return _DATA_CACHE[task]


def run_seed(task, kind, s, batch, device, threads=1, timing=False):
    """One initialisation seed of one cell: a plain dict (picklable, so it can
    run in a worker process). Seed ``s`` fully determines the run, so the
    result does not depend on which process runs it or in what order."""
    torch.set_num_threads(threads)
    _, epochs, lr = TASKS[task]
    Xtr, Ytr, Xte, Yte, classify = task_data(task)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    if Xte is not None:
        Xte, Yte = Xte.to(device), Yte.to(device)
    n_in, n_out = Xtr.shape[1], Ytr.shape[1]
    t0 = time.perf_counter()
    model, scaler = train(kind, n_in, n_out, SPLIT_SEED + 7919 * (s + 1), Xtr, Ytr,
                          epochs, lr, batch, device)
    r = {"train_s": time.perf_counter() - t0}
    native = MODELS[kind][0] == "c"
    if native:
        Xtr, Ytr = Xtr.cpu(), Ytr.cpu()
        if Xte is not None:
            Xte, Yte = Xte.cpu(), Yte.cpu()
    r["mse"], r["acc"] = metrics(model, Xtr, Ytr, classify)
    r["hold_mse"], r["hold_acc"] = (metrics(model, Xte, Yte, classify) if Xte is not None
                                    else (None, None))
    r["params"] = model.num_params()
    r["layers"] = getattr(model, "depth", 2 if scaler is None else model.depth)
    r["structure"] = model.structure() if hasattr(model, "structure") else None
    if scaler is not None:
        r["counters"] = dict(vars(scaler.counters))
    elif hasattr(model, "counters"):
        r["counters"] = model.counters()
    else:
        r["counters"] = {}
    r["us_per_infer"] = us_per_infer(model, Xtr) if timing else None
    r["n_in"], r["n_out"] = n_in, n_out
    return r


def aggregate(task, kind, runs, batch, device):
    """The board row of one cell from its per-seed runs (in seed order)."""
    _, epochs, lr = TASKS[task]

    def mean(v):
        v = [x for x in v if x is not None]
        return statistics.fmean(v) if v else None

    def sd(v):
        v = [x for x in v if x is not None]
        return statistics.stdev(v) if len(v) > 1 else 0.0

    col = {k: [r[k] for r in runs] for k in runs[0] if k not in ("counters",)}
    family, rule, through = MODELS[kind]
    n_in, n_out = runs[0]["n_in"], runs[0]["n_out"]
    out = {"impl": IMPL.get(kind, f"torch-{kind}"), "task": task, "rule": rule,
           "backend": "nativetypenn" if family == "c" else "torch",
           "width_through_depth_probe": through, "device": str(device),
           "seeds": len(runs), "epochs": epochs, "lr": lr, "batch_size": batch,
           "hold_mse": mean(col["hold_mse"]), "hold_mse_sd": sd(col["hold_mse"]),
           "hold_acc": mean(col["hold_acc"]), "hold_acc_sd": sd(col["hold_acc"]),
           "mse": mean(col["mse"]), "acc": mean(col["acc"]),
           "params": mean(col["params"]), "params_sd": sd(col["params"]),
           "train_s": mean(col["train_s"]), "us_per_infer": runs[-1]["us_per_infer"],
           "init_layers": TypeNN(n_in, n_out).birth_depth if family in ("type-nn", "c") else 2,
           "structure": col["structure"] if any(col["structure"]) else None,
           "layers": mean(col["layers"]),
           "per_seed_hold_mse": col["hold_mse"]}
    for k in ("or_add", "or_drop", "and_add", "and_drop", "layer_add", "layer_drop"):
        out[k] = mean([r["counters"].get(k, 0) for r in runs])
    return out


def cell(task, kind, seeds, batch, device, verbose=False):
    runs = []
    for s in range(seeds):
        runs.append(run_seed(task, kind, s, batch, device, torch.get_num_threads(),
                             timing=s + 1 == seeds))
        if verbose:
            r = runs[-1]
            print(f"  {task} {kind} seed {s}: hold_mse {r['hold_mse']} train_mse {r['mse']:.2e}"
                  f" params {r['params']} layers {r['layers']} {r['train_s']:.1f}s",
                  file=sys.stderr)
    return aggregate(task, kind, runs, batch, device)


def _job(args):
    task, kind, s, batch, device, timing = args
    return run_seed(task, kind, s, batch, device, 1, timing)


def cells_parallel(tasks, kinds, seeds, batch, device, jobs, verbose=False):
    """Every (task, model, seed) in its own single-threaded worker; rows come
    back in the serial order. Wall times are measured under load (all cores
    busy), so compare ``train_s`` only within one run."""
    import concurrent.futures as cf
    import multiprocessing as mp

    todo = [(t, k, s, batch, device, s + 1 == seeds)
            for t in tasks for k in kinds for s in range(seeds)]
    # the slowest cells first, so the pool drains evenly
    order = sorted(range(len(todo)), key=lambda i: -TASKS[todo[i][0]][1]
                   * (1 if MODELS[todo[i][1]][0] == "mlp" else 10))
    ctx = mp.get_context("spawn")        # CUDA and forked workers do not mix
    results: dict[int, dict] = {}
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        futs = {ex.submit(_job, todo[i]): i for i in order}
        for f in cf.as_completed(futs):
            i = futs[f]
            results[i] = f.result()
            if verbose:
                t, k, s = todo[i][:3]
                print(f"  done {t} {k} seed {s} ({len(results)}/{len(todo)})", file=sys.stderr)
    i = 0
    for t in tasks:
        for k in kinds:
            yield aggregate(t, k, [results[i + s] for s in range(seeds)], batch, device)
            i += seeds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", nargs="?", default="all",
                    help="all, or a comma-separated list of: " + ", ".join(TASKS))
    ap.add_argument("model", nargs="?", default="all",
                    help="all, or a comma-separated list of: " + ", ".join(MODELS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes, one (task, model, seed) each; 0 = all cores")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    tasks = list(TASKS) if a.task == "all" else a.task.split(",")
    if bad := [t for t in tasks if t not in TASKS]:
        ap.error(f"unknown task(s) {bad}; choose from {list(TASKS)}")
    kinds = list(MODELS) if a.model == "all" else a.model.split(",")
    unknown = [k for k in kinds if k not in MODELS]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; choose from {list(MODELS)}")
    jobs = a.jobs or os.cpu_count() or 1
    if jobs > 1:
        for row in cells_parallel(tasks, kinds, a.seeds, a.batch_size, a.device, jobs,
                                  a.verbose):
            print(json.dumps(row), flush=True)
        return
    for task in tasks:
        for kind in kinds:
            print(json.dumps(cell(task, kind, a.seeds, a.batch_size, a.device, a.verbose)),
                  flush=True)


if __name__ == "__main__":
    main()
