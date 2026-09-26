# Audit

An audit of the library and its benchmark for honesty, fairness and
consistency with the architecture, and what was done about each finding.
The source of truth is the architecture as briefed by its author:

- a layer is a partition function, `z_k = sign(A_k) ln(1 + |A_k|)`,
  `A_k = prod_r (w_r . x + b_r)^(a_kr)`; each layer's output is the next
  layer's input, as dense as an MLP;
- all scaling happens in back-propagation: a width (Or) probe is a new
  output of the previous layer, trained, and read by the current layer
  through a dummy weight; an And carries a noisy identity Or and gets a new
  one when back-prop gives it weight; a layer is inserted between *any* two
  layers;
- scaling down happens by thresholds: an element whose weight is below the
  threshold is dropped, of two identity Ors in an And one is dropped, a layer
  that became an identity (up to the threshold) is dropped;
- a network is born with `ln(1 + n m)` layers; scale up early, drop late.

Git: `baseline` is the tree as received; each iteration is one commit.

## What held up

- **The math.** Partition function, signed power, log-space forward, the
  one-sided backward at an exact zero factor, and per-Or Adam match the C
  reference to 1e-12 (`tests/test_crosscheck_c.py`).
- **The structure edits** (width/degree/depth probes, the depth fold, birth
  depth, the grow/fit/prune schedule) are a line-by-line port of
  `type_nn_scale.c`.
- **The harness.** `bench.py` matches `bench.c` (split, train-only
  standardisation, shuffle, `lr × 0.1`, the MLP width rule and init law).
  The hold-out never reaches the scaler.
- **The numbers.** Every figure in the old BOARD.md matched the raw result
  files it cited; nothing was fabricated.
- **The data.** The five UCI/NCSU files were byte-identical to the type-nn
  flake's pins; `digits.csv.gz` to scikit-learn 1.5.2's copy.

## Findings

| # | finding | kind | resolution | tested by |
|---|---|---|---|---|
| 1 | Only C's BIC rule was ported. Its promotions and pruning re-run training pairs with forward passes; the briefed rule decides from back-prop displacement and thresholds alone (C's `type-nn-overfit`), and was missing, as was that model's row on the board. | spec | `rule="threshold"` ported and made the default; `rule="bic"` kept; `ref-type-nn-overfit` on the board against C | `tests/test_threshold.py` (a test fails if the rule re-reads any pair), the board's port section |
| 2 | Width was blocked on junctions touching the depth probe (C's rule), so a network born at depth ≤ 2 could not grow width while the probe held the gaps; the MLP adapter did not block, so the two families ran different rules. | spec, consistency | width on every junction by default for both; `width_through_depth_probe=False` reproduces C | `test_width_grows_at_birth_depth_two_by_default` |
| 3 | The board compared torch type-nn with C's MLP, never with the torch MLP of the same harness. | fairness | like-for-like section (same harness, framework, seeds); port section compares each C-exact torch model with its C model | `board.py` sections |
| 4 | Friedman #1 was generated in code with torch's RNG (not a fixed dataset; can change between torch versions); its quoted noise floor (0.00101) was computed by no code. | honesty, reproducibility | the fixed Delve/OpenML 564 "fried" file, pinned by hash; the floor `1/span²` computed and reported per run | `test_friedman_file_is_friedman_one` checks the file *is* Friedman #1 (y − f(x) has mean ≈ 0, variance ≈ 1) |
| 5 | The width option was introduced after seeing Friedman results and then judged "best model on the task" on the same hold-out. | honesty | the claim is withdrawn; width on every junction is now the default because the architecture says so, not because of a result | BOARD.md "Notes on honesty" |
| 6 | Dropping several coordinates of one junction folded only the first one's mean into the consumer (a drop resets the consumer's statistics). Shared with C. | bug | all means taken before any drop | `test_every_coordinate_dropped_on_a_junction_keeps_its_mean` (old path: error 4e-4) |
| 7 | `StructureScaler.end()` before the prune phase measured on no pairs, so the BIC criterion saw only the parameter count and pruned everything prunable. Shared with C; `fit` never took the path. | bug | the last epoch's pairs are kept until the next step | `test_end_before_the_prune_phase_measures_on_real_pairs` (old path: 86 → 12 params at 2.3× the MSE) |
| 8 | With a stock `torch.optim` optimizer nothing enforced `a ≥ 1` (an assembly index drifted to 0.41); found by the new device tests. | bug | `keep_invariants` post-step hook, attached by the scaler | `test_type_nn_with_a_stock_optimizer_on_the_device`, `test_edits.py` |
| 9 | The datasets were committed to the repository. | reproducibility | pinned in `benchmarks/datasets.json`, fetched by the flake (`fetchurl`) or `fetch_data.py`, git-ignored; the flake checks run the dataset tests with the pinned files | `tests/test_datasets.py` (hashes, shapes, formats) |
| 10 | `scale.py` built its models separately from the board, had no `--device`, and defaulted to 3 seeds (the board: 5). | consistency | models from `bench.build`; `--device`; 5 seeds | `test_every_benchmark_model_trains_on_the_device` |
| 11 | No GPU coverage beyond one test; `nix flake check` did not test CUDA. | coverage | every path parametrised over `cpu`/`cuda` (`tests/test_device.py`); `checks.cuda` under `nix flake check --impure` when a GPU and the `cuda` system feature are present; `nix run .#test-cuda` on the host | run on an RTX 5070 Ti: 117 passed, 0 skipped |
| 12 | BOARD.md's tables were edited by hand. | honesty | every table generated by `board.py` from named result files; pre-audit results archived and not read | `board.py --write` is idempotent |
| 13 | Width prune listed the depth probe as a producer. With width-through-probe (the default) the same coordinate was then a candidate twice: dropping it on the probe made the identity layer non-square and desynchronised `out_features`/`in_features` down the stack, so the model was no longer as dense as an MLP. `params_without` also kept counting the probe's carrier, so the BIC trial size was wrong. The MLP adapter already skipped the probe. | spec, consistency, bug | `prune_candidates` / `commit_removals` never treat the probe as a width producer; `params_without` drops the carrier when the real producer goes | `test_width_candidates_skip_the_depth_probe`, `test_params_without_counts_the_probe_carrier` |
| 14 | The torch `AndOr` pads every unit to max degree (`mask` + identity slots), so memory grows with `m·R·n` rather than with the live Ors — the opposite of the architecture's footprint goal. | spec | torch path kept for batched GEMMs; C `type-nn` and `type-nn-overfit` wrapped as `NativeTypeNN` (ragged store); `get_backend("cuda")` reserved for a device store with the same live-Or layout | `tests/test_native.py`, `docs/BACKENDS.md` |

## Upstream issues met on the way (not in this code)

- nixpkgs: `cudaPackages_13` (13.3) applies a cccl patch the 13.3.3 release
  already contains, so cccl fails to build. The flake pins
  `cudaPackages_13_0`, the CUDA the torch 2.13 wheel is built for.
- NVIDIA re-pointed NCCL's `v2.32.3-1` tag (2026-09-17), so nixpkgs' pinned
  hash no longer matches what GitHub serves. The flake pins the tag's current
  commit by rev with an independently computed hash, through nixpkgs'
  `_cuda.extensions` hook, and only while nixpkgs still has the stale hash.
