"""Does a graph port take C's decisions? A diagnostic for the graph board rows.

    python benchmarks/graph_parity.py [task] [--seeds 0,1,2,3,4] [--port cuda|torch]

For each board seed (bench.py's ``SPLIT_SEED + 7919 (s + 1)``, its split,
shuffle, epochs and lr) it runs the C reference (NativeGraphTypeNN) and a port
side by side and reports:

- **free run**: the first epoch whose structure or counters differ from C's,
  with the output gap at that point, and the final train MSE of both. Per-sample
  training is chaotic, so a 1e-16 rounding difference (another torch build,
  another BLAS) can grow until a decision flips; a free-run divergence alone is
  not a bug.
- **re-sync** (``--port cuda`` only): every epoch, a fresh port is loaded from
  C's own state (``CudaGraphTypeNN.from_snapshot``) and runs that one epoch.
  Any mismatch here is a decision the port takes differently *from the same
  state*: a rule difference, or an exact tie that rounding breaks the other
  way. Those are what to report.

Exit status 1 if any re-sync epoch mismatched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

from torch_type_nn import (  # noqa: E402
    CudaGraphTypeNN,
    GraphScaler,
    GraphTypeNN,
    NativeGraphTypeNN,
    TypeAdam,
)
from torch_type_nn.graph.core import xorshift32  # noqa: E402


def shuffle_seed(ep: int) -> int:
    return bench.SPLIT_SEED ^ (((ep + 1) * 0x9E3779B9) & bench.MASK)


class TorchRun:
    """GraphTypeNN driven by bench.c's protocol (batch 1, C shuffle)."""

    def __init__(self, n_in, n_out, seed, n, epochs, lr):
        self.m = GraphTypeNN(n_in, n_out, seed=seed)
        self.opt = TypeAdam(self.m, lr=lr * 0.1)          # bench.c: task lr x 0.1
        self.sc = GraphScaler(self.m, self.opt, epochs=epochs, steps_per_epoch=n, n_train=n)
        self.sc.begin()

    def epoch(self, X, Y, seed):
        n = X.shape[0]
        order, s = list(range(n)), seed
        for i in range(n, 1, -1):
            s = xorshift32(s)
            j = s % i
            order[i - 1], order[j] = order[j], order[i - 1]
        self.m.train()
        for i in order:
            y = self.m(X[i:i + 1])
            loss = torch.nn.functional.mse_loss(y, Y[i:i + 1])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.sc.observe(X[i:i + 1], y, Y[i:i + 1])
        self.sc.epoch_end()

    def end(self):
        self.sc.end()

    def structure(self):
        return self.m.structure()

    def counters(self):
        return self.m.counters()

    def __call__(self, X):
        with torch.no_grad():
            return self.m.eval()(X)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="xor", choices=list(bench.TASKS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--port", default="cuda", choices=["cuda", "torch"])
    ap.add_argument("--no-resync", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    print(f"torch {torch.__version__}, python {sys.version.split()[0]}, "
          f"task {a.task}, port {a.port}")
    _, E, lr = bench.TASKS[a.task]
    Xtr, Ytr, *_ = bench.task_data(a.task)
    n_in, n_out, n = Xtr.shape[1], Ytr.shape[1], Xtr.shape[0]
    bad_total = 0
    for si in (int(s) for s in a.seeds.split(",")):
        seed = bench.SPLIT_SEED + 7919 * (si + 1)
        c = NativeGraphTypeNN(n_in, n_out, seed=seed)
        c.begin(n, E, lr)
        if a.port == "cuda":
            p = CudaGraphTypeNN(n_in, n_out, seed=seed, device="cpu")
            p.begin(n, E, lr)
        else:
            p = TorchRun(n_in, n_out, seed, n, E, lr)
        first, resync_bad = None, 0
        for ep in range(E):
            s = shuffle_seed(ep)
            q = (None if a.no_resync or a.port != "cuda"
                 else CudaGraphTypeNN.from_snapshot(c.snapshot(), device="cpu"))
            c.epoch(Xtr, Ytr, s)
            p.epoch(Xtr, Ytr, s)
            if q is not None:
                q.epoch(Xtr, Ytr, s)
                if q.structure() != c.structure() or q.counters() != c.counters():
                    resync_bad += 1
                    if resync_bad <= 3:
                        print(f"  seed {si} RE-SYNC MISMATCH epoch {ep} "
                              f"(phase {c.scalars()['phase']})")
                        print(f"    C    {c.structure()} {c.counters()}")
                        print(f"    port {q.structure()} {q.counters()}")
            if first is None and (p.structure() != c.structure()
                                  or p.counters() != c.counters()):
                first = ep
                gap = float((c(Xtr) - p(Xtr)).abs().max())
                print(f"  seed {si} free run leaves C at epoch {ep} "
                      f"(phase {c.scalars()['phase']}, output gap {gap:.2e})")
                print(f"    C    {c.structure()} {c.counters()}")
                print(f"    port {p.structure()} {p.counters()}")
        c.end()
        p.end()
        mc = float(((c(Xtr) - Ytr) ** 2).mean())
        mp = float(((p(Xtr) - Ytr) ** 2).mean())
        print(f"seed {si}: train MSE C {mc:.2e} port {mp:.2e}; "
              f"free run {'= C throughout' if first is None else f'left C at epoch {first}'}; "
              + ("re-sync skipped" if a.no_resync or a.port != "cuda"
                 else f"re-sync mismatches {resync_bad}/{E}"), flush=True)
        bad_total += resync_bad
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
