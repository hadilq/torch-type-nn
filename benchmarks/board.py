"""Markdown board: torch-type-nn against the reference C type-nn and its MLP baseline.

    python benchmarks/board.py benchmarks/out/torch-board.jsonl \
        [--reference benchmarks/reference/c-type-nn-7333bf7.jsonl] > BOARD.md

A hold-out gap is "clear" when it exceeds two standard errors of the
difference (the rule the C board uses); otherwise it is within noise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
ORDER = ["xor", "iris", "wine", "wdbc", "diabetes", "ionosphere"]
NAMES = {"type-nn": "C type-nn", "c-mlp": "C c-mlp",
         "torch-type-nn": "torch-type-nn", "torch-mlp": "torch MLP",
         "torch-mlp-scaled": "torch MLP, scaled"}


def read(paths):
    rows = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["task"], r["impl"])] = r
    return rows


def verdict(a, b):
    """a vs b on hold-out MSE: (difference, 2se, clear?)."""
    if a.get("hold_mse") is None or b.get("hold_mse") is None:
        return None
    se = math.sqrt(a["hold_mse_sd"] ** 2 / a["seeds"] + b["hold_mse_sd"] ** 2 / b["seeds"])
    d = a["hold_mse"] - b["hold_mse"]
    return d, 2 * se, abs(d) > 2 * se


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--reference", default=str(HERE / "reference" / "c-type-nn-7333bf7.jsonl"))
    a = ap.parse_args(argv)
    rows = read([a.reference, *a.results])
    impls = [i for i in NAMES if any(k[1] == i for k in rows)]

    out = ["| task | impl | hold MSE | ± | hold acc | train MSE | params | ± | layers "
           "| train s | µs/inf |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    notes = []
    for t in ORDER:
        for i in impls:
            r = rows.get((t, i))
            if not r:
                continue
            out.append(f"| {t} | {NAMES[i]} | {fmt(r['hold_mse'], '.4f')} "
                       f"| {fmt(r['hold_mse_sd'] if r['hold_mse'] is not None else None, '.4f')} "
                       f"| {fmt(r['hold_acc'], '.3f')} | {r['mse']:.2e} | {r['params']:.1f} "
                       f"| {r['params_sd']:.1f} | {r['init_layers']}→{r['layers']:.1f} "
                       f"| {r['train_s']:.2f} | {r['us_per_infer']:.1f} |")
        tt = rows.get((t, "torch-type-nn"))
        ct, cm = rows.get((t, "type-nn")), rows.get((t, "c-mlp"))
        ms, mf = rows.get((t, "torch-mlp-scaled")), rows.get((t, "torch-mlp"))
        pairs = [(tt, ct, "torch-type-nn", "C type-nn"), (tt, cm, "torch-type-nn", "C c-mlp"),
                 (ms, mf, "torch MLP, scaled", "torch MLP (fixed)")]
        for tt, other, me, label in pairs:
            if tt and other:
                v = verdict(tt, other)
                ratio = tt["params"] / other["params"]
                if v is None:
                    notes.append(f"- **{t}**, {me} vs {label}: fit only; "
                                 f"params {ratio:.2f}×")
                else:
                    d, se2, clear = v
                    word = ("lower" if d < 0 else "higher") + (", clear" if clear
                                                              else ", within noise")
                    notes.append(f"- **{t}**, {me} vs {label}: hold MSE {word} "
                                 f"({d:+.4f}, 2se {se2:.4f}); params {ratio:.2f}×")
    print("\n".join(out))
    print()
    print("\n".join(notes))


if __name__ == "__main__":
    main()
