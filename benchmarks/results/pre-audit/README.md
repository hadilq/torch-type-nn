# Pre-audit results (archived)

Produced before the audit (commit `baseline` in git) and **not comparable with
the current board**; kept for the record only. `board.py` does not read them.

What changed since, and affects these numbers:

- the default rule is now the threshold rule (these used the BIC rule);
- width grows on every junction by default (these: the C width rule, except
  `*-through*`);
- `end()` before the prune phase, mean compensation on multi-coordinate drops,
  and `a >= 1` with stock optimizers were fixed;
- Friedman #1 here was drawn in `scale.py` with torch's RNG (10 000 rows), not
  the pinned Delve/OpenML file (40 768 rows); its quoted noise floor was not
  computed by any code.

Current results: `../board-per-sample.jsonl` and `../scale-batch32.jsonl`,
written by `nix run .#board-all`.
