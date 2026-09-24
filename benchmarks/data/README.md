# Benchmark data

The files are **not committed** (`.gitignore` excludes everything here but
this README). They are pinned by URL and SRI hash in
[`../datasets.json`](../datasets.json), the manifest that both `flake.nix` and
`fetch_data.py` read:

    nix develop                          # fetches through the Nix store, copies here
    nix run .#data                       # the same copy, without a shell
    python benchmarks/fetch_data.py      # without Nix: download + sha256 check
    python benchmarks/fetch_data.py --verify-only

`nix run .#bench`, `nix run .#scale` and `nix flake check` read the store copy
directly (`TORCH_TYPE_NN_DATA`); a hash mismatch fails the fetch.

| file               | task       | source                                                             | license   |
| ------------------ | ---------- | ------------------------------------------------------------------ | --------- |
| `iris.data`        | iris       | UCI Machine Learning Repository, Iris (Fisher, 1936)               | CC BY 4.0 |
| `wine.data`        | wine       | UCI, Wine (Aeberhard & Forina, 1991)                               | CC BY 4.0 |
| `wdbc.data`        | wdbc       | UCI, Breast Cancer Wisconsin (Diagnostic) (Wolberg et al., 1995)   | CC BY 4.0 |
| `ionosphere.data`  | ionosphere | UCI, Ionosphere (Sigillito et al., 1989)                           | CC BY 4.0 |
| `diabetes.tab.txt` | diabetes   | Efron, Hastie, Johnstone & Tibshirani (2004), "Least Angle Regression", via NCSU | public research data |
| `digits.csv.gz`    | digits     | UCI Optical Recognition of Handwritten Digits (Alpaydin & Kaynak, 1998), the 1797-sample 8x8 set as shipped in scikit-learn 1.5.2 | CC BY 4.0 |
| `564_fried.tsv.gz` | friedman   | Friedman #1 (Friedman, 1991; Breiman, 1996), 40 768 rows, as distributed by Delve / OpenML 564 "fried" and mirrored by PMLB (pinned commit) | public research data |

The five board files and their hashes are those of the type-nn reference
flake (byte-identical). Friedman #1 used to be generated in `scale.py` with
torch's RNG, which is not a fixed dataset (the stream can change between torch
versions); it is now this fixed file, and `tests/test_datasets.py` checks that
it really is Friedman #1: inputs in [0, 1], and `y - f(x)` zero-mean with unit
variance.
