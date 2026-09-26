# Differences from the C reference

torch-type-nn is a port of [hadilq/type-nn](https://github.com/hadilq/type-nn)
at commit `7333bf7`. The math is the same and is checked number for number
(`tests/test_crosscheck_c.py`: forward, every gradient, and two per-Or Adam
steps agree to 1e-12 on ragged layers). Both C scaling rules are ported:
`rule="bic"` is `type_nn_scale.c` (C's `type-nn`) and `rule="threshold"` is
`type_nn_overfit_scale.c` (C's `type-nn-overfit`); the defaults follow the
architecture (threshold rule, width on every junction), and
`TypeNNAdapter(width_through_depth_probe=False)` with either rule is the
exact C configuration (`ref-*` on the board). Where the port behaves
differently, it is listed here.

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

## 5. Width on every junction (default)

The reference grows width only on junctions that do not touch the depth
probe (`tnn_grow_width` skips a pair when either layer is the probe). A
network born with depth `D <= 2` (`round(ln(1 + n m)) <= 2`, i.e. `n m <= 11`)
then has no width site while the depth probe occupies the only gaps: two live
layers plus the probe give two junctions, and both touch it. Under C's BIC
rule the probe rarely gets promoted on such networks, and the C board shows
the effect: xor (born 1) and diabetes (born 2) are the only tasks with zero
width promotions. Under C's threshold rule a promoted probe adds a third live
layer and with it a junction away from the probe, so xor does grow width
there, while diabetes still does not.

The architecture grows width between any two layers, so the port does so by
default: a width probe on a junction that crosses the depth probe adds the
unit to the producer, a carrier Or for it to the probe layer (`w = e_k`,
`b = 0`, `a = 1`: the probe layer stays square and identity-like), and a
zero column to the real consumer beyond; the edit is exact because the
consumer reads the unit with weights of exactly 0. Displacement, evidence
and removal are measured at the real consumer. The identity layer also
carries an existing width probe instead of retiring it when inserted.
`width_through_depth_probe=False` reproduces C.

Prune lists a width item only on the producing layer (the depth probe is a
carrier, not a producer). C's `prune_width` walks every consecutive pair,
including the probe as a producer; that is harmless when width is blocked
next to the probe, and is still what `width_through_depth_probe=False` does
for a junction whose *consumer* is the probe. Listing the probe as a
producer under the default rule would drop the same coordinate twice and
break the dense stack.

## 6. The threshold rule's band with batches

C's `type-nn-overfit` drops items inside `theta_band = lr N^(3/4)` with `N`
the training samples, i.e. the optimizer steps per epoch of its per-sample
training. The port uses `S`, the optimizer steps per epoch (`N` when
`batch_size=1`), consistent with `theta(T)` counting optimizer steps (§3).
The order is C's: the depth probe (retired inside the band, else promoted)
or at most one near-identity live layer, then width, then degree, each judged
on the structure the previous axis left; a junction keeps its most-moved
coordinate and an And its most-moved Or (the first one on a tie, as C's
`keep_max`).

## 7. Fixes that C does not have

- **Mean compensation on several drops at one junction.** Dropping a
  coordinate folds its mean into the consumer's bias (`b += w E[x_k]`), but
  dropping an input also resets the consumer's statistics, so in C only the
  first coordinate dropped at a junction in one boundary gets its mean; the
  others are dropped without it. That matters under the threshold rule
  (under the BIC rule the dropped columns were ablated to 0 first, so there
  is nothing to fold). The port takes every mean before any drop.
- **`end()` before the prune phase** (BIC rule). The epoch's pairs used to be
  released at the boundary, so an `end()` called before the prune phase
  measured on no pairs: the criterion saw only the parameter count and pruned
  everything prunable. The pairs now stay until the next step. `fit` never
  took this path (the last boundary always falls in the prune phase).
- **`a >= 1` with stock optimizers.** `TypeAdam` projects the assembly index
  after every step; a stock `torch.optim` optimizer did not, so `a` could
  drift below 1. The scaler now attaches `keep_invariants` to any optimizer.
