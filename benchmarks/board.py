"""Render the tables of BOARD.md from benchmark results; no number is typed by hand.

    python benchmarks/board.py [--board results/board-per-sample.jsonl]
                               [--scale results/scale-batch32.jsonl]
                               [--reference reference/c-type-nn-7333bf7.jsonl]
                               [--write BOARD.md]

Without ``--write`` the sections are printed. With it, each section replaces
the text between its markers in BOARD.md (``<!-- board:NAME -->`` ...
``<!-- /board:NAME -->``); the prose around them is left alone. A section
whose inputs are missing says so instead of keeping old numbers.

A hold-out gap is *clear* when it exceeds two standard errors of the
difference, ``2 sqrt(sd_a^2/n_a + sd_b^2/n_b)`` (the rule the C board uses);
otherwise it is *within noise*.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
ORDER = ["xor", "iris", "wine", "wdbc", "diabetes", "ionosphere"]
SCALE_ORDER = ["digits", "friedman"]

# impl id -> display name, in table order
NAMES = {
    "type-nn": "C type-nn",
    "type-nn-overfit": "C type-nn-overfit",
    "c-mlp": "C c-mlp",
    "torch-ref-type-nn": "torch ref-type-nn",
    "torch-ref-type-nn-overfit": "torch ref-type-nn-overfit",
    "torch-type-nn": "**torch type-nn**",
    "torch-type-nn-bic": "torch type-nn-bic",
    "torch-mlp": "torch MLP",
    "torch-mlp-scaled": "torch MLP, scaled",
}
SCALE_NAMES = {"type-nn": "**type-nn**", "type-nn-bic": "type-nn-bic",
               "mlp-scaled": "MLP, scaled", "mlp-16": "MLP-16", "mlp-64": "MLP-64"}

# (a, b, why): a against b on hold-out MSE
PORT = [("torch-ref-type-nn", "type-nn", "port of C type-nn (BIC rule, C width rule)"),
        ("torch-ref-type-nn-overfit", "type-nn-overfit",
         "port of C type-nn-overfit (threshold rule, C width rule)")]
LIKE = [("torch-type-nn", "torch-mlp", "the architecture vs the MLP baseline"),
        ("torch-type-nn-bic", "torch-mlp", "with the BIC prior vs the MLP baseline"),
        ("torch-type-nn", "torch-type-nn-bic", "threshold rule vs BIC rule"),
        ("torch-mlp-scaled", "torch-mlp", "the rule on an MLP vs the fixed MLP")]
CBOARD = [("type-nn", "c-mlp", "C board"), ("type-nn-overfit", "c-mlp", "C board")]
SCALE_PAIRS = [("type-nn", "mlp-16"), ("type-nn", "mlp-64"), ("type-nn-bic", "mlp-64"),
               ("type-nn", "type-nn-bic"), ("mlp-scaled", "mlp-16")]

MISSING = "_Not generated yet: run `nix run .#board-all` (or see the commands above)._"


def read(paths):
    rows, found = {}, []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        found.append(p)
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["task"], r["impl"])] = r
    return rows, found


def verdict(a, b):
    """(difference, 2se, clear) on hold-out MSE, or None without a hold-out."""
    if a.get("hold_mse") is None or b.get("hold_mse") is None:
        return None
    se = math.sqrt(a["hold_mse_sd"] ** 2 / a["seeds"] + b["hold_mse_sd"] ** 2 / b["seeds"])
    d = a["hold_mse"] - b["hold_mse"]
    return d, 2 * se, abs(d) > 2 * se


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def rel(paths):
    out = []
    for p in paths:
        try:
            out.append(f"`{Path(p).resolve().relative_to(ROOT)}`")
        except ValueError:
            out.append(f"`{p}`")
    return ", ".join(out)


def per_sample_table(rows):
    impls = [i for i in NAMES if any(k[1] == i for k in rows)]
    out = ["| task | model | hold MSE | ± sd | hold acc | train MSE | params | ± sd "
           "| layers | train s | µs/inf |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in ORDER:
        for i in impls:
            r = rows.get((t, i))
            if not r:
                continue
            sd = r["hold_mse_sd"] if r["hold_mse"] is not None else None
            out.append(
                f"| {t} | {NAMES[i]} | {fmt(r['hold_mse'], '.4f')} | {fmt(sd, '.4f')} "
                f"| {fmt(r['hold_acc'], '.3f')} | {fmt(r['mse'], '.2e')} | {r['params']:.1f} "
                f"| {r['params_sd']:.1f} | {r['init_layers']}→{r['layers']:.1f} "
                f"| {fmt(r.get('train_s'), '.2f')} | {fmt(r.get('us_per_infer'), '.1f')} |")
    return out


def pairs_table(rows, pairs, names, order):
    out = ["| task | comparison | Δ hold MSE | 2se | verdict | params ratio |",
           "|---|---|---:|---:|---|---:|"]
    n = 0
    for t in order:
        for a, b, *_ in pairs:
            ra, rb = rows.get((t, a)), rows.get((t, b))
            if not (ra and rb):
                continue
            n += 1
            label = f"{names[a]} vs {names[b]}"
            ratio = ra["params"] / rb["params"] if rb["params"] else float("nan")
            v = verdict(ra, rb)
            if v is None:
                fit = "both fit" if (ra.get("acc") == 1 and rb.get("acc") == 1) else (
                    f"train acc {fmt(ra.get('acc'), '.2f')} vs {fmt(rb.get('acc'), '.2f')}")
                out.append(f"| {t} | {label} | n/a | n/a | {fit} (no hold-out) | {ratio:.2f}× |")
                continue
            d, se2, clear = v
            word = ("lower" if d < 0 else "higher") + (", **clear**" if clear
                                                        else ", within noise")
            out.append(f"| {t} | {label} | {d:+.4f} | {se2:.4f} | {word} | {ratio:.2f}× |")
    return out if n else None


def scale_table(rows):
    impls = [i for i in SCALE_NAMES if any(k[1] == i for k in rows)]
    out = ["| task | model | hold MSE | ± sd | × noise floor | hold acc | train MSE | params "
           "| layers | train s |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in SCALE_ORDER:
        for i in impls:
            r = rows.get((t, i))
            if not r:
                continue
            floor = r.get("noise_floor")
            mult = r["hold_mse"] / floor if floor else None
            out.append(
                f"| {t} | {SCALE_NAMES[i]} | {r['hold_mse']:.5f} | {r['hold_mse_sd']:.5f} "
                f"| {fmt(mult, '.2f')} | {fmt(r.get('hold_acc'), '.3f')} "
                f"| {fmt(r.get('train_mse'), '.5f')} | {r['params']:.0f} "
                f"| {fmt(r.get('layers'), '.1f')} | {fmt(r.get('train_s'), '.1f')} |")
    floors = {t: rows[(t, i)].get("noise_floor") for (t, i) in rows
              if rows[(t, i)].get("noise_floor")}
    for t, f in sorted(floors.items()):
        out.append("")
        out.append(f"{t}: noise floor (best reachable hold-out MSE, `1 / span²`) = {f:.5f}")
    return out


def meta(rows):
    ks = {(r.get("seeds"), r.get("batch_size"), r.get("device", "cpu")) for r in rows.values()
          if r["impl"].startswith("torch-") or r["impl"] in SCALE_NAMES}
    return "; ".join(f"{s} seeds, batch {b}, {d}" for s, b, d in sorted(ks, key=str))


def sections(board, scale, reference):
    ref_rows, ref_found = read([reference])
    b_rows, b_found = read(board)
    rows = {**ref_rows, **b_rows}
    s_rows, s_found = read(scale)
    src = f"Generated by `benchmarks/board.py` from {rel(ref_found + b_found)}"
    out = {}
    have_torch = bool(b_rows)
    table = per_sample_table(rows)
    out["per-sample"] = [f"{src} ({meta(b_rows) or 'C reference only'}).", "", *table]
    if not have_torch:
        out["per-sample"] += ["", MISSING]
    for name, pairs in (("port", PORT), ("like-for-like", LIKE), ("c-board", CBOARD)):
        t = pairs_table(rows, pairs, NAMES, ORDER)
        out[name] = t if t else [MISSING]
    if s_rows:
        pairs = pairs_table(s_rows, [(a, b) for a, b in SCALE_PAIRS], SCALE_NAMES, SCALE_ORDER)
        out["scale"] = [f"Generated by `benchmarks/board.py` from {rel(s_found)} "
                        f"({meta(s_rows)}).", "", *scale_table(s_rows), "",
                        *(pairs or [])]
    else:
        out["scale"] = [MISSING]
    return out


def write(path, secs):
    text = Path(path).read_text()
    for name, lines in secs.items():
        n = re.escape(name)
        pat = re.compile(rf"(<!-- board:{n} -->\n).*?(<!-- /board:{n} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"{path}: no <!-- board:{name} --> section")
        body = "\n".join(lines)
        text = pat.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
    Path(path).write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", nargs="*",
                    default=[str(HERE / "results" / "board-per-sample.jsonl")])
    ap.add_argument("--scale", nargs="*", default=[str(HERE / "results" / "scale-batch32.jsonl")])
    ap.add_argument("--reference", default=str(HERE / "reference" / "c-type-nn-7333bf7.jsonl"))
    ap.add_argument("--write", metavar="BOARD.md")
    a = ap.parse_args(argv)
    secs = sections(a.board, a.scale, a.reference)
    if a.write:
        write(a.write, secs)
    else:
        for name, lines in secs.items():
            print(f"<!-- board:{name} -->\n" + "\n".join(lines) + f"\n<!-- /board:{name} -->\n")


if __name__ == "__main__":
    main()
