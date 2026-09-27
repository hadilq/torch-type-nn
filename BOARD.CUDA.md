# CUDA board

Numbers on this page come from `benchmarks/board.py --write-cuda`. They are
**not** bit-identical to C `bench.c`; C stays the number-for-number backend
on [BOARD.md](BOARD.md). This file is the ragged device store
(`CudaTypeNN`: packed `W (R, n_in)`, fused `X @ W.T`).

`nix flake check --impure` skips `checks.cuda` unless the Nix daemon enables
the `cuda` system feature. A GPU machine without that feature still runs the
suite on the host:

```sh
nix run .#test-cuda          # pytest + xor on the device (prints the probe)
nix run .#board-cuda         # 5-seed CUDA board + scale, then this file
```

or step by step:

```sh
nix run .#bench-cuda -- all cuda-type-nn,cuda-type-nn-overfit,mlp \
    --seeds 5 --device cuda --jobs 0 \
    > benchmarks/results/board-cuda.jsonl
nix run .#scale-cuda -- all --models cuda-type-nn,cuda-type-nn-overfit,mlp-16,mlp-64 \
    --seeds 5 --device cuda --jobs 0 \
    > benchmarks/results/scale-cuda.jsonl
nix run .#board -- \
    --board benchmarks/results/board-cuda.jsonl \
    --scale benchmarks/results/scale-cuda.jsonl \
    --write-cuda BOARD.CUDA.md
```

Look for `======== torch-type-nn CUDA probe ========` and
`torch-type-nn device: CUDA …` in the log. A green run with only
`torch-type-nn device: CPU` did not use a GPU.

## Per-sample tasks

Same tasks, split, epochs and lr as [BOARD.md](BOARD.md). CUDA rows are
per-sample Adam on the device (C protocol, fused GEMV). MLP rows in the
same jsonl were trained with `--device cuda`.

<!-- board:per-sample -->
_Not generated yet: run `nix run .#board-cuda`._
<!-- /board:per-sample -->

## Comparisons

Hold-out gap is **clear** when it exceeds two standard errors of the
difference, `2 sqrt(sd_a²/n_a + sd_b²/n_b)`.

<!-- board:pairs -->
_Not generated yet: run `nix run .#board-cuda`._
<!-- /board:pairs -->

## Larger data

Same `scale.py` tasks as [BOARD.md](BOARD.md) (digits, Friedman #1). CUDA
rows use `CudaTypeNN.epoch` (per-sample on the device). MLP-16 / MLP-64 in
this file are the batched baseline on the same `--device`.

<!-- board:scale -->
_Not generated yet: run `nix run .#board-cuda`._
<!-- /board:scale -->
