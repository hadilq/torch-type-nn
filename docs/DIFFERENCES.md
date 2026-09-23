# Differences from the C reference

torch-type-nn is a port of [hadilq/type-nn](https://github.com/hadilq/type-nn)
at commit `7333bf7`. The math is the same and is checked number for number
(`tests/test_crosscheck_c.py`: forward, every gradient, and two per-Or Adam
steps agree to 1e-12 on ragged layers). Where the port behaves differently,
it is listed here.

## 1. Pruning counts parameters exactly

`prune_pool` ablates candidates greedily and keeps an ablation while the
criterion `C = n ln MSE + K ln n` stays at or below the best seen while
pruning. The README of type-nn defines `K` as *all parameters*.

The C code does not recompute `K`; it subtracts a per-candidate tally from
the starting count:

```c
if (tnn_criterion(net, m, P - K - x->np) > ref) { /* restore */ }
```

The tally counts some parameters twice. An Or ablated on its own is
counted with `x->np = n_in + 2`, and if its unit's coordinate is ablated
later, `coordinate_params` counts all of that unit's Ors again. The
trial model is therefore scored as smaller than it is, so the rule
accepts ablations the stated criterion would reject.

In C, `P`, `K` and `np` are `size_t`. Once the tally exceeds `P`, the
subtraction wraps to a huge number, the criterion becomes enormous, and the
candidate is rejected. This accidental guard stops the worst cases. Python
integers do not wrap, and a literal port collapsed an iris network from 198
parameters to 21 at 50× the training MSE (the fixture
`tests/data/iris_prune_boundary.pt`, used by
`test_prune_boundary_respects_the_exact_criterion`).

The port computes the exact parameter count of each trial model
(`(live Ors) × (live inputs + 2)` per surviving unit). With it, every prune
boundary satisfies the invariant that `test_prune_never_worse` in the C
suite states: the criterion never rises above `max(before, best)`.

This is likely worth an upstream issue: before the wrap, the C prune is
more aggressive than its documented rule.

## 2. Initialisation RNG

C draws every random number from one `xorshift32` stream. The port draws
from a `torch.Generator` owned by the model (`TypeNN.generator`), so a
given seed does not reproduce C's weights. The benchmark therefore compares
means over 5 seeds, not single runs. The split, standardisation and
per-epoch shuffle of the benchmark *do* use C's `xorshift32` and are
identical (row for row) to `bench.c`.

## 3. Batches

C trains per sample. The port also trains per sample by default (`fit(...,
batch_size=1)`), and everything type-nn defines stays well defined with
batches:

- the step count (for `theta(T)` and the schedule) counts optimizer steps;
- the evidence rule re-evaluates every sample seen in the epoch;
- the depth statistics accumulate per sample; the mean `|dL/dx|` is scaled
  by `1/B` under a batch-mean loss, which does not change which gap is
  loudest.

The board uses `batch_size=1` so it matches the C protocol.

## 4. Storage

Layers are dense padded tensors (`(m, R, n)`) instead of linked arrays of
Ors. Free slots hold the identity Or, which is exactly 1, and get no
gradient. Dropped Ors leave a free slot until `compact()` removes slot
columns that are free in every unit (done at every prune boundary and at
the end of training). None of this changes the function.

## 5. Width growth next to the depth probe (opt-in change)

The reference grows width only on junctions that do not touch the depth
probe (`tnn_grow_width` skips a pair when either layer is the probe). A
network born with depth `D <= 2` (`round(ln(1 + n m)) <= 2`, i.e. `n m <= 11`)
then has **no** width site as soon as the first depth probe is inserted: two
live layers plus the probe give two junctions, and both touch it. The depth
probe is only moved, never removed, during growth, so width stays blocked
for the whole grow phase.

The C board shows it: the two tasks born at depth <= 2 are the only ones
with zero width promotions (`or_add`): xor (born 1) and diabetes (born 2).
Every task born at depth >= 3 grows width. On diabetes this is harmless (the
data are small and nearly linear); on Friedman #1 (10 inputs, 1 output,
born at depth 2, 7000 training rows) it caps type-nn at width 1 and a
hold-out MSE several times the noise floor (`benchmarks/scale.py`).

`TypeNNAdapter(model, width_through_depth_probe=True)` lifts the block. A
width probe on a junction that crosses the depth probe adds the unit to the
producer, a carrier Or for it to the probe layer (`w = e_k`, `b = 0`,
`a = 1`: the probe layer stays square and identity-like), and a zero column
to the real consumer beyond; the edit is exact because the consumer reads
the unit with weights of exactly 0. Displacement, evidence and removal are
measured at the real consumer. With the option on, the identity layer also
carries an existing width probe instead of retiring it when inserted.

The default keeps the reference behaviour, so the validated port still
reproduces C; the effect of the option is measured in BOARD.md.
