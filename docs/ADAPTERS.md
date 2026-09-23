# Scaling other architectures: writing an adapter

`StructureScaler` implements type-nn's rule and nothing else:

- **Grow** (first third of training, while the residual is unexplained): on
  each axis a probe is an exact identity; a probe that moved past
  `theta(T) = lr T^(3/4)` *and* whose ablation costs more than its BIC
  price (`n ln(MSE_without / MSE_with) > k ln n`) is promoted, and a fresh
  probe takes its place.
- **Prune** (last third): the depth probe, then at most one live layer, is
  removed on trial; then every width and degree item is ablated alone,
  sorted by the damage it does, and added greedily to a joint ablation
  while the criterion `n ln MSE + K ln n` (exact `K`) stays at or below the
  best seen while pruning; the ablated set is then removed for real.

What an *item* is comes from an adapter implementing
`torch_type_nn.protocol.Scalable`. `TypeNNAdapter`
(`src/torch_type_nn/adapters/typenn.py`) is the reference.

## The contract

| method | meaning | must be |
|---|---|---|
| `probe_sites(axis)` | where probes live (junctions, units, ...) | stable order |
| `probe_at(axis, site)` / `add_probe(axis, site, ctx)` | find / insert a probe | the new probe is an **exact identity** up to `ctx.noise` |
| `promote(item)` | probe becomes live | |
| `born`, `displacement` | age and RMS distance from identity | `inf` if the item has no identity |
| `best_site(DEPTH, moving)`, `site_of`, `remove_probe` | where depth is wanted; move the probe | |
| `ablate(item)` | reset to identity in place, return undo | **undo restores bit for bit** |
| `cost(item)` | parameters the item owns | |
| `trial_remove(item)` | remove a layer tentatively | `undo` restores exactly |
| `prune_candidates`, `can_remove`, `params_without`, `commit_removals` | the pool | `params_without` is the exact count; `commit_removals` of ablated items does not change the function |
| `set_edit_sink(list)` | report parameter replacements as `Edit`s | needed for stock optimizers |

An axis an architecture does not have simply returns no sites and no
candidates.

## Identities by architecture

| axis | type-nn | MLP (`Linear` + activation) | residual nets / transformers |
|---|---|---|---|
| width | new unit, next layer reads it with weight 0 | same (Net2Net widening) | same; attention head with zero output projection |
| degree | Or with w = 0, b = 1 (factor 1) | none | gates starting at 1 |
| depth | identity layer + affine fold of `F` | exact only in special cases | residual block with zero last projection: `x + 0` |

## Adapters shipped

- `TypeNNAdapter` (`adapters/typenn.py`): width, degree, depth; the port of
  `type_nn_scale.c`.
- `MLPAdapter` (`adapters/mlp.py`) for `ScalableMLP`
  (`Linear -> ReLU -> ... -> Linear [-> F]`): width and depth.
  - Width: a new hidden unit read with an all-zero column (exact).
  - Depth: `ReLU(I a + 0) = a` because post-ReLU activations are >= 0, so
    an identity layer is exact in any gap *after* a ReLU; no fold is needed.
    There is no exact identity in front of the first layer, so that gap is
    never used.
  - A depth probe does not block width growth. type-nn skips junctions
    touching the depth probe; with one hidden layer that would leave an MLP
    no width site at all (the first version of this adapter grew depth and
    never width). Instead, a width probe on a junction that crosses the
    depth probe passes through it on an identity entry (weight exactly 1):
    the probe layer stays square and exact, and displacement, evidence and
    removal are measured at the real consumer beyond it.

## Optimizers

With `TypeAdam` every Or keeps its own step count and edits move its
state exactly. With any stock `torch.optim` optimizer the scaler records the
adapter's `Edit`s during each boundary and calls `follow_structure`: state
tensors shaped like a parameter are remapped along the edit's index map
(new slices start at 0), new parameters are added, removed ones dropped.
Two approximations remain with stock optimizers: new slices inherit their
tensor's scalar step count (Adam's bias correction), and weight rescaling
done by depth folds is not applied to the optimizer's moments.

## Testing an adapter

The type-nn adapter is held to: bit-identical golden runs across the
refactor, `test_probes_are_exact_identities`, the prune-invariant fixture
test, and the C cross-check. A new adapter should at least show that probes
are identities, that `ablate`/undo round-trips exactly, that
`commit_removals` preserves the function, and that a prune boundary never
raises the exact criterion.
