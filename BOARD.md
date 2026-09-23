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
| iris | C type-nn | 0.0215 | 0.0069 | 0.969 | 8.43e-04 | 125.4 | 52.2 | 3→3.8 | 0.48 | 0.6 |
| iris | C c-mlp | 0.0280 | 0.0061 | 0.956 | 1.09e-02 | 67.0 | 0.0 | 2→2.0 | 0.02 | 0.1 |
| iris | torch-type-nn | 0.0265 | 0.0059 | 0.960 | 3.02e-03 | 124.8 | 48.8 | 3→4.0 | 52.77 | 145.9 |
| iris | torch MLP | 0.0309 | 0.0043 | 0.956 | 1.48e-02 | 67.0 | 0.0 | 2→2.0 | 10.80 | 26.2 |
| wine | C type-nn | 0.0277 | 0.0048 | 0.956 | 7.85e-05 | 169.8 | 52.3 | 4→5.0 | 0.56 | 0.7 |
| wine | C c-mlp | 0.0234 | 0.0056 | 0.974 | 2.24e-03 | 275.0 | 0.0 | 2→2.0 | 0.03 | 0.2 |
| wine | torch-type-nn | 0.0215 | 0.0140 | 0.967 | 3.25e-05 | 208.8 | 44.2 | 4→5.0 | 60.16 | 175.8 |
| wine | torch MLP | 0.0259 | 0.0032 | 0.956 | 3.87e-03 | 275.0 | 0.0 | 2→2.0 | 10.35 | 26.6 |
| wdbc | C type-nn | 0.0448 | 0.0048 | 0.951 | 1.01e-04 | 162.8 | 28.7 | 3→4.0 | 0.20 | 0.4 |
| wdbc | C c-mlp | 0.0438 | 0.0017 | 0.946 | 5.18e-03 | 513.0 | 0.0 | 2→2.0 | 0.11 | 0.3 |
| wdbc | torch-type-nn | 0.0447 | 0.0080 | 0.951 | 3.51e-04 | 175.8 | 36.3 | 3→4.0 | 62.31 | 139.6 |
| wdbc | torch MLP | 0.0471 | 0.0040 | 0.944 | 5.70e-03 | 513.0 | 0.0 | 2→2.0 | 13.48 | 26.7 |
| diabetes | C type-nn | 0.0329 | 0.0010 | n/a | 2.73e-02 | 20.4 | 7.2 | 2→2.4 | 0.07 | 0.2 |
| diabetes | C c-mlp | 0.0488 | 0.0042 | n/a | 1.35e-02 | 193.0 | 0.0 | 2→2.0 | 0.05 | 0.2 |
| diabetes | torch-type-nn | 0.0328 | 0.0005 | n/a | 2.73e-02 | 18.6 | 4.9 | 2→2.2 | 73.39 | 81.3 |
| diabetes | torch MLP | 0.0478 | 0.0038 | n/a | 1.47e-02 | 193.0 | 0.0 | 2→2.0 | 21.84 | 28.1 |
| ionosphere | C type-nn | 0.0886 | 0.0111 | 0.904 | 9.70e-03 | 153.8 | 65.0 | 4→4.4 | 0.28 | 0.1 |
| ionosphere | C c-mlp | 0.1461 | 0.0155 | 0.896 | 1.63e-02 | 577.0 | 0.0 | 2→2.0 | 0.08 | 0.4 |
| ionosphere | torch-type-nn | 0.0895 | 0.0223 | 0.904 | 5.58e-03 | 188.8 | 16.8 | 4→5.2 | 76.27 | 188.1 |
| ionosphere | torch MLP | 0.1356 | 0.0128 | 0.908 | 1.57e-02 | 577.0 | 0.0 | 2→2.0 | 13.43 | 27.8 |

- **xor**, torch-type-nn vs C type-nn: fit only; params 0.94×
- **xor**, torch-type-nn vs C c-mlp: fit only; params 0.75×
- **iris**, torch-type-nn vs C type-nn: hold MSE higher, within noise (+0.0050, 2se 0.0082); params 1.00×
- **iris**, torch-type-nn vs C c-mlp: hold MSE lower, within noise (-0.0015, 2se 0.0076); params 1.86×
- **wine**, torch-type-nn vs C type-nn: hold MSE lower, within noise (-0.0062, 2se 0.0132); params 1.23×
- **wine**, torch-type-nn vs C c-mlp: hold MSE lower, within noise (-0.0019, 2se 0.0135); params 0.76×
- **wdbc**, torch-type-nn vs C type-nn: hold MSE lower, within noise (-0.0001, 2se 0.0083); params 1.08×
- **wdbc**, torch-type-nn vs C c-mlp: hold MSE higher, within noise (+0.0009, 2se 0.0073); params 0.34×
- **diabetes**, torch-type-nn vs C type-nn: hold MSE lower, within noise (-0.0002, 2se 0.0010); params 0.91×
- **diabetes**, torch-type-nn vs C c-mlp: hold MSE lower, clear (-0.0160, 2se 0.0038); params 0.10×
- **ionosphere**, torch-type-nn vs C type-nn: hold MSE higher, within noise (+0.0010, 2se 0.0223); params 1.23×
- **ionosphere**, torch-type-nn vs C c-mlp: hold MSE lower, clear (-0.0565, 2se 0.0243); params 0.33×

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
- The MLP baseline differs slightly between C and torch (initialisation RNG);
  both are shown.

Raw results: `benchmarks/results/`. Reproduce: `nix run .#bench -- all all --seeds 5 > out.jsonl && nix run .#board -- out.jsonl`.
