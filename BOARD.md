# Board: torch-type-nn vs the C reference

Same machine (1 CPU core), same protocol: `bench.c` ported line for line
(`benchmarks/bench.py`); identical split, standardisation and shuffle;
per-sample Adam at `lr × 0.1`; 5 initialisation seeds per cell (mean, ± is
the standard deviation). C reference: hadilq/type-nn `7333bf7`
(`benchmarks/reference/`). A gap is *clear* only above two standard errors.

| task | impl | hold MSE | ± | hold acc | train MSE | params | ± | layers | train s | µs/inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xor | C type-nn | n/a | n/a | n/a | 5.87e-09 | 26.4 | 3.6 | 1→2.0 | 0.02 | 0.1 |
| xor | C c-mlp | n/a | n/a | n/a | 2.30e-07 | 33.0 | 0.0 | 2→2.0 | 0.00 | 0.0 |
| xor | torch-type-nn | n/a | n/a | n/a | 8.81e-07 | 24.8 | 1.8 | 1→2.0 | 12.38 | 72.4 |
| xor | torch MLP | n/a | n/a | n/a | 7.71e-06 | 33.0 | 0.0 | 2→2.0 | 3.49 | 28.2 |
| xor | torch MLP, scaled | n/a | n/a | n/a | 8.34e-02 | 13.0 | 8.4 | 2→2.6 | 7.93 | 33.4 |
| iris | C type-nn | 0.0215 | 0.0069 | 0.969 | 8.43e-04 | 125.4 | 52.2 | 3→3.8 | 0.48 | 0.6 |
| iris | C c-mlp | 0.0280 | 0.0061 | 0.956 | 1.09e-02 | 67.0 | 0.0 | 2→2.0 | 0.02 | 0.1 |
| iris | torch-type-nn | 0.0265 | 0.0059 | 0.960 | 3.02e-03 | 124.8 | 48.8 | 3→4.0 | 52.77 | 145.9 |
| iris | torch MLP | 0.0309 | 0.0043 | 0.956 | 1.48e-02 | 67.0 | 0.0 | 2→2.0 | 10.80 | 26.2 |
| iris | torch MLP, scaled | 0.0293 | 0.0079 | 0.956 | 7.94e-03 | 49.2 | 15.8 | 2→3.0 | 22.12 | 32.9 |
| wine | C type-nn | 0.0277 | 0.0048 | 0.956 | 7.85e-05 | 169.8 | 52.3 | 4→5.0 | 0.56 | 0.7 |
| wine | C c-mlp | 0.0234 | 0.0056 | 0.974 | 2.24e-03 | 275.0 | 0.0 | 2→2.0 | 0.03 | 0.2 |
| wine | torch-type-nn | 0.0215 | 0.0140 | 0.967 | 3.25e-05 | 208.8 | 44.2 | 4→5.0 | 60.16 | 175.8 |
| wine | torch MLP | 0.0259 | 0.0032 | 0.956 | 3.87e-03 | 275.0 | 0.0 | 2→2.0 | 10.35 | 26.6 |
| wine | torch MLP, scaled | 0.0191 | 0.0112 | 0.967 | 2.65e-03 | 105.4 | 13.6 | 2→3.0 | 21.42 | 34.2 |
| wdbc | C type-nn | 0.0448 | 0.0048 | 0.951 | 1.01e-04 | 162.8 | 28.7 | 3→4.0 | 0.20 | 0.4 |
| wdbc | C c-mlp | 0.0438 | 0.0017 | 0.946 | 5.18e-03 | 513.0 | 0.0 | 2→2.0 | 0.11 | 0.3 |
| wdbc | torch-type-nn | 0.0447 | 0.0080 | 0.951 | 3.51e-04 | 175.8 | 36.3 | 3→4.0 | 62.31 | 139.6 |
| wdbc | torch MLP | 0.0471 | 0.0040 | 0.944 | 5.70e-03 | 513.0 | 0.0 | 2→2.0 | 13.48 | 26.7 |
| wdbc | torch MLP, scaled | 0.0558 | 0.0087 | 0.942 | 1.76e-03 | 124.0 | 23.5 | 2→3.0 | 27.07 | 33.9 |
| diabetes | C type-nn | 0.0329 | 0.0010 | n/a | 2.73e-02 | 20.4 | 7.2 | 2→2.4 | 0.07 | 0.2 |
| diabetes | C c-mlp | 0.0488 | 0.0042 | n/a | 1.35e-02 | 193.0 | 0.0 | 2→2.0 | 0.05 | 0.2 |
| diabetes | torch-type-nn | 0.0328 | 0.0005 | n/a | 2.73e-02 | 18.6 | 4.9 | 2→2.2 | 73.39 | 81.3 |
| diabetes | torch MLP | 0.0478 | 0.0038 | n/a | 1.47e-02 | 193.0 | 0.0 | 2→2.0 | 21.84 | 28.1 |
| diabetes | torch MLP, scaled | 0.0372 | 0.0009 | n/a | 2.59e-02 | 21.2 | 5.5 | 2→2.2 | 37.87 | 25.0 |
| ionosphere | C type-nn | 0.0886 | 0.0111 | 0.904 | 9.70e-03 | 153.8 | 65.0 | 4→4.4 | 0.28 | 0.1 |
| ionosphere | C c-mlp | 0.1461 | 0.0155 | 0.896 | 1.63e-02 | 577.0 | 0.0 | 2→2.0 | 0.08 | 0.4 |
| ionosphere | torch-type-nn | 0.0895 | 0.0223 | 0.904 | 5.58e-03 | 188.8 | 16.8 | 4→5.2 | 76.27 | 188.1 |
| ionosphere | torch MLP | 0.1356 | 0.0128 | 0.908 | 1.57e-02 | 577.0 | 0.0 | 2→2.0 | 13.43 | 27.8 |
| ionosphere | torch MLP, scaled | 0.1308 | 0.0382 | 0.870 | 1.98e-02 | 119.4 | 48.0 | 2→3.0 | 25.34 | 32.1 |

- **xor**, torch-type-nn vs C type-nn: fit only; params 0.94×
- **xor**, torch-type-nn vs C c-mlp: fit only; params 0.75×
- **xor**, torch MLP, scaled vs torch MLP (fixed): fit only; params 0.39×
- **iris**, torch-type-nn vs C type-nn: hold MSE higher, within noise (+0.0050, 2se 0.0082); params 1.00×
- **iris**, torch-type-nn vs C c-mlp: hold MSE lower, within noise (-0.0015, 2se 0.0076); params 1.86×
- **iris**, torch MLP, scaled vs torch MLP (fixed): hold MSE lower, within noise (-0.0016, 2se 0.0081); params 0.73×
- **wine**, torch-type-nn vs C type-nn: hold MSE lower, within noise (-0.0062, 2se 0.0132); params 1.23×
- **wine**, torch-type-nn vs C c-mlp: hold MSE lower, within noise (-0.0019, 2se 0.0135); params 0.76×
- **wine**, torch MLP, scaled vs torch MLP (fixed): hold MSE lower, within noise (-0.0067, 2se 0.0104); params 0.38×
- **wdbc**, torch-type-nn vs C type-nn: hold MSE lower, within noise (-0.0001, 2se 0.0083); params 1.08×
- **wdbc**, torch-type-nn vs C c-mlp: hold MSE higher, within noise (+0.0009, 2se 0.0073); params 0.34×
- **wdbc**, torch MLP, scaled vs torch MLP (fixed): hold MSE higher, clear (+0.0086, 2se 0.0086); params 0.24×
- **diabetes**, torch-type-nn vs C type-nn: hold MSE lower, within noise (-0.0002, 2se 0.0010); params 0.91×
- **diabetes**, torch-type-nn vs C c-mlp: hold MSE lower, clear (-0.0160, 2se 0.0038); params 0.10×
- **diabetes**, torch MLP, scaled vs torch MLP (fixed): hold MSE lower, clear (-0.0105, 2se 0.0035); params 0.11×
- **ionosphere**, torch-type-nn vs C type-nn: hold MSE higher, within noise (+0.0010, 2se 0.0223); params 1.23×
- **ionosphere**, torch-type-nn vs C c-mlp: hold MSE lower, clear (-0.0565, 2se 0.0243); params 0.33×
- **ionosphere**, torch MLP, scaled vs torch MLP (fixed): hold MSE lower, within noise (-0.0048, 2se 0.0360); params 0.21×

## Reading it

- **The port matches C.** On every task with a hold-out, torch-type-nn and C
  type-nn are within noise of each other on hold-out MSE, with a similar
  parameter count (0.91–1.23×). XOR fits on every seed in both.
- **C's findings reproduce.** Against the MLP baseline, type-nn is clearly
  better on diabetes (with 0.10× the parameters) and ionosphere (0.33×), and
  within noise on iris, wine and wdbc: the same verdicts as the C board.
- **Speed.** Per-sample training in Python is dispatch-bound (about 2 ms
  per step against C's microseconds). The batched path is the fast one:
  iris at `--batch-size 16` trains in 9.1 s per seed instead of 52 s, with
  hold-out MSE 0.0288 ± 0.0020 (acc 0.956), keeping more structure (264
  params, 5 layers), since fewer optimizer steps mean a lower displacement
  threshold.
- The fixed MLP baseline differs slightly between C and torch (initialisation RNG);
  both are shown.

Raw results: `benchmarks/results/`. Reproduce: `nix run .#bench -- all all --seeds 5 > out.jsonl && nix run .#board -- out.jsonl`.

## The rule on an MLP (step 6)

`torch MLP, scaled` is `ScalableMLP` under `MLPAdapter`: born with one
hidden layer of `max(2, m)` units, trained with stock `torch.optim.Adam` on
the same protocol as the fixed `torch MLP` (8 or 16 hidden units, chosen by
the C board's author), with width and depth grown and pruned by type-nn's
rule. Final hidden widths per seed are in the raw results.

- **Size.** The scaled MLP ends at 0.11–0.73× the fixed MLP's parameters
  on every task (diabetes: 21 against 193).
- **Hold-out.** Clearly better on diabetes (0.0372 against 0.0478, the same
  verdict type-nn gets there), within noise on iris, wine and ionosphere,
  and clearly worse on wdbc (+0.0086, exactly at the 2se line).
- **XOR fails.** Two of five seeds prune to a single hidden ReLU unit,
  which cannot represent XOR (train MSE 0.083, against 8e-6 fixed). With
  4 samples the BIC evidence needs a probe to cut the MSE about 4× before
  it pays for 4 parameters, so width barely grows, and pruning is cheap.
  type-nn fits XOR with one product unit; a ReLU MLP needs two, and the
  rule does not find the second. This is the degenerate end of the
  criterion (n = 4), but it is a real failure.
- **Growth is conservative.** About 1–1.6 width promotions and one depth
  promotion per run: the final networks have at most 5 units per hidden
  layer. On these tasks that is enough; on larger problems it would
  likely under-grow, since one probe per junction per epoch and a BIC
  price per unit make growth slow by design.
- **Wall time** is about 2× the fixed MLP (statistics, evidence passes).

So on this board the rule transfers: it finds much smaller MLPs of similar
quality, with one clear win, one clear loss and one failure on a
degenerate task. That is evidence, not a verdict; the next test is larger
data.


## Larger data (step 7)

`benchmarks/scale.py`, mini-batches of 32, stock Adam at 0.003 (TypeAdam for
type-nn), 5 seeds, same 70/30 xorshift split. **digits**: UCI optical
digits, 1797 x 64 pixels, 10 classes, 60 epochs. **friedman**: Friedman #1,
10 000 x 10 inputs (5 pure noise), 40 epochs; the hold-out MSE of the true
function (the noise floor) is **0.00101**.

| task | model | hold MSE | ± | hold acc | params | train s |
|---|---|---:|---:|---:|---:|---:|
| digits | type-nn (reference rule) | 0.00734 | 0.00189 | 0.957 | 3517 | 55.7 |
| digits | type-nn, width through depth probe | 0.00723 | 0.00085 | 0.956 | 3393 | 55.0 |
| digits | MLP, scaled | 0.00989 | 0.00149 | 0.957 | 1007 | 2.5 |
| digits | MLP, fixed 16 | 0.01110 | 0.00074 | 0.959 | 1210 | 1.2 |
| digits | MLP, fixed 64 | 0.01089 | 0.00022 | 0.974 | 4810 | 1.3 |
| friedman | type-nn (reference rule) | 0.00473 | 0.00203 | n/a | 47 | 16.1 |
| friedman | type-nn, width through depth probe | 0.00115 | 0.00002 | n/a | 125 | 17.0 |
| friedman | MLP, scaled | 0.00149 | 0.00015 | n/a | 92 | 7.9 |
| friedman | MLP, fixed 16 | 0.00127 | 0.00004 | n/a | 193 | 4.0 |
| friedman | MLP, fixed 64 | 0.00128 | 0.00009 | n/a | 769 | 4.3 |

- **The reference rule under-grows on Friedman.** Born at depth 2, type-nn
  has no width site once its depth probe exists (see
  [docs/DIFFERENCES.md §5](docs/DIFFERENCES.md)): 0.4 width promotions per
  run, width 1, hold-out 4.7x the noise floor.
- **With `width_through_depth_probe=True` it is the best model on
  Friedman**: 0.00115 on every seed (14% above the floor), clearly better
  than the fixed 16-unit MLP (-0.00012, 2se 0.00004) with 0.65x its
  parameters, and than the 64-unit MLP with 0.16x.
- **Digits: MSE and accuracy disagree.** type-nn has the lowest hold-out
  MSE by a clear margin (0.0072 against 0.0109 for the 64-unit MLP), but
  the 64-unit MLP has the best accuracy (0.974 against 0.956). The rule
  selects structure by MSE, the loss it is trained on; for classification
  a likelihood criterion (`2 NLL + K ln n` with cross-entropy) is the
  natural next step. Digits is born at depth 6, so the new option changes
  nothing there (within noise).
- **The scaled MLP**: within noise of the fixed MLP on digits with 0.83x
  the parameters; clearly worse on Friedman (0.00149 against 0.00127) with
  0.48x. It stops growing early on this task.
- **Cost.** type-nn trains 4x (Friedman) to 40x (digits) slower than an MLP here
  (per-step Python overhead of the padded layer plus evidence passes).

### The option on the original board

Per-sample protocol, 5 seeds, `type-nn-through` against the reference rule
(`benchmarks/results/torch-board-per-sample-through.jsonl`):

| task | hold MSE, reference → option | verdict | params | width promotions |
|---|---|---|---|---|
| xor | fits → fits (identical) | | 24.8 → 24.8 | 0 → 0 |
| iris | 0.0265 → 0.0228 | within noise | 125 → 130 | 1.0 → 2.2 |
| wine | 0.0215 → 0.0241 | within noise | 209 → 216 | 1.8 → 3.0 |
| wdbc | 0.0447 → 0.0499 | within noise | 176 → 211 | 1.4 → 3.2 |
| diabetes | 0.0328 → 0.0330 | within noise | 19 → 18 | 0.0 → 0.6 |
| ionosphere | 0.0895 → 0.1007 | within noise | 189 → 164 | 2.4 → 3.4 |

No task changes beyond noise; width grows more on every task, and the
blocked case (Friedman) goes from the worst model to the best.
