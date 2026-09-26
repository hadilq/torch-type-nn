# torch-type-nn

A PyTorch port of [type-nn](https://github.com/hadilq/type-nn): a neural
network whose layers are partition functions of an And over Ors, and which
grows and prunes its own width, degree and depth while it trains. The
design is explained in
[Train the knowledge](https://hadilq.com/posts/train-the-knowledge/).

> Status: alpha (0.1.0.dev0). Forward, gradients and the per-Or Adam step
> agree with the C reference to 1e-12, and both C scaling rules have exact
> ports on the board (see [Validation](#validation)). Deliberate differences
> from C are listed in [docs/DIFFERENCES.md](https://github.com/hadilq/torch-type-nn/blob/main/docs/DIFFERENCES.md); the audit
> that led to the current defaults is in [docs/AUDIT.md](https://github.com/hadilq/torch-type-nn/blob/main/docs/AUDIT.md).

## Quick start

```python
import torch
from torch_type_nn import TypeNN, fit

model = TypeNN(in_features=4, out_features=3)     # birth: round(ln(1 + n m)) layers
result = fit(model, X, Y, epochs=250, lr=0.005)   # grows and prunes while it trains
print(model.structure(), model.num_params(), result.counters)
```

`fit` reproduces type-nn's per-sample protocol by default
(`batch_size=1`); pass `batch_size=64` (or more) for the batched, GPU-friendly
path. There is no width, degree or depth to choose.

For your own training loop, drive the structure with a `StructureScaler`:

```python
from torch_type_nn import TypeNN, TypeAdam, StructureScaler

model = TypeNN(n, m)
opt = TypeAdam(model, lr=lr)
scaler = StructureScaler(model, opt, epochs=E, steps_per_epoch=len(loader))
scaler.begin()
for epoch in range(E):
    for x, t in loader:
        y = model(x)
        loss = 0.5 * (y - t).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        scaler.observe(x, y, t)          # residual statistics (and pairs, for rule="bic")
    scaler.epoch_end()                   # grow / prune at the boundary
scaler.end()                             # last prune; no probe survives
```

### C backend (ragged Ors, smaller footprint)

The torch `AndOr` pads every unit to the layer's max degree so a batch is
one GEMM. That uses more memory than the live graph. The C models keep one
`w[n_in]` per live Or and scale memory with the structure. They are wrapped
as `NativeTypeNN` (compiled on first use; needs `cc`):

```python
from torch_type_nn import NativeTypeNN, available_backends, get_backend

net = NativeTypeNN(4, 3, rule="threshold")   # C type-nn-overfit
# net = NativeTypeNN(4, 3, rule="bic")       # C type-nn
result = net.fit(X, Y, epochs=250, lr=0.05)  # task lr; C applies × 0.1
print(net.structure(), net.num_params(), result.counters)
```

`get_backend("torch")` / `get_backend("c")` return the class.
`get_backend("cuda")` is reserved for a device store with the same ragged
layout — not a port of the padded tensors. See
[docs/BACKENDS.md](docs/BACKENDS.md).

## Structure learning

One dummy rule on three axes. A **probe** is an identity that back-prop is
free to move; a probe that back-prop moved past `theta(T) = lr T^(3/4)`
(`T` its age in optimizer steps) is promoted, and a fresh probe takes its
place. All scaling up happens early and all scaling down late.

| axis   | probe (grow, `u < 1/3`)                                           | drop (prune, `u >= 2/3`)                       |
| ------ | ----------------------------------------------------------------- | ---------------------------------------------- |
| width (Or)  | the previous layer grows a trained unit; the current layer reads it with a dummy weight of exactly 0 | the coordinate, when its weights are back inside the band (a junction keeps its most-moved one) |
| degree (And)| each And carries one noisy identity Or (w ~ 0, b ~ 1)       | an Or back inside the band (an And keeps its most-moved Or: of two identity Ors one goes) |
| depth  | an identity layer in the gap, between *any* two layers, with the largest mean `|dL/dx|` | a layer back to identity within the band (one per boundary; folded) |

(`u = step / total`; the band is `theta_band = lr S^(3/4)`, `S` optimizer
steps per epoch.) Growth is gated on an unexplained residual
(`MSE > Var(t) / N`). Depth edits fold the best affine fit between `x` and
`F(x)` into the next layer, so they are near-exact. A network is born with
`round(ln(1 + n m))` layers, `n` inputs and `m` outputs, each layer's output
feeding the next, as dense as an MLP.

**Two rules.** `StructureScaler(..., rule=...)`:

- `"threshold"` (default) is the architecture: every decision comes from
  what back-prop did to the dummy weights; no training pair is stored or
  re-run. It is the rule of C's `type-nn-overfit`.
- `"bic"` adds the Bayesian-information prior of the post: a moved probe is
  promoted only if it also pays `n ln(MSE_without / MSE_with) > k ln n`,
  where `MSE_without` is *measured* by resetting the item to its identity
  and re-running the epoch's training pairs (so the scaler keeps one epoch
  of `(x, t)` in memory), and pruning ablates items while the criterion
  `n ln MSE + K ln n` allows. It is the rule of C's `type-nn`.

**Width on every junction.** By default a width probe also grows on a
junction that crosses the depth probe (carried through the probe layer).
C blocks width there, which leaves a network born with depth ≤ 2 no width
site while a depth probe exists; `TypeNNAdapter(model,
width_through_depth_probe=False)` reproduces C.

Both rules and both width rules are on [BOARD.md](https://github.com/hadilq/torch-type-nn/blob/main/BOARD.md), including exact
ports of the two C models.

**Any architecture, any optimizer.** The scaler is written against a
small protocol (`torch_type_nn.protocol.Scalable`): the rule lives in
`StructureScaler`, and what an item *is* (how to insert an identity, measure
its distance from it, remove it) comes from an adapter. `TypeNN` is wrapped
in `TypeNNAdapter` automatically. `ScalableMLP` (`Linear`/ReLU stacks) is
the second family: `MLPAdapter` grows and prunes its width and depth. See
[docs/ADAPTERS.md](https://github.com/hadilq/torch-type-nn/blob/main/docs/ADAPTERS.md) for writing an adapter. Stock
`torch.optim` optimizers work too: structural edits are reported as `Edit`s
with index maps, `follow_structure` remaps the optimizer state through them,
and the scaler attaches `keep_invariants` so `a >= 1` holds after every step.

## The layer

```python
import torch
from torch_type_nn import AndOr, TypeAdam

layer = AndOr(in_features=8, out_features=4, ors_per_unit=2)
z = layer(torch.randn(32, 8))          # (32, 4)
```

Unit k of a layer is one And over its Ors r:

```
Or_kr = w_kr . x + b_kr                sum type
A_k   = prod_r Or_kr ^ a_kr            product type, a_kr >= 1
z_k   = sign(A_k) ln(1 + |A_k|)        partition function
```

- **Batched and padded.** All Ors of a layer live in one `(m, R, n)`
  tensor; a unit with fewer than `R` Ors has identity Ors (w = 0, b = 1,
  a = 1) in its free slots. The identity is exactly 1, so padding never
  changes the output, and its gradients are masked so no optimizer moves
  it. This is what gives type-nn batching and GPU support.
- **Exact backward.** `AndOrFunction` computes the gradients in log
  space, including the one-sided rule at a factor that is exactly 0
  (cofactor slope when a = 1, flat when a > 1), where autograd through
  `log|O|` would give NaN.
- **Per-Or Adam.** `TypeAdam(model, lr)` gives every Or its own step
  count, so an Or born late gets the same bias correction as one born at
  the start, and keeps `a >= 1`. It takes the module, not a parameter list,
  because structural edits replace the parameter tensors; the Adam moments
  live in the layer (`adam_*` buffers) so edits move them with the
  weights. Non-type-nn parameters in the same model get ordinary Adam.
  Stock `torch.optim` optimizers also work on a layer of fixed structure.
- **Structural primitives.** `add_or`, `drop_or`, `add_unit`,
  `drop_unit`, `add_input` (a zero column: function preserved exactly),
  `drop_input`, `compact`. `load_state_dict` accepts a checkpoint of a
  different structure.

## Validation

Two independent checks against [hadilq/type-nn](https://github.com/hadilq/type-nn):

- **Number for number.** `tests/test_crosscheck_c.py` compiles the C code
  (`tests/c/crosscheck.c`), builds the same ragged network in both, and
  compares the forward pass, dL/dx, every dL/dw, dL/db and dL/da, and the
  weights after two per-Or Adam steps. Everything agrees to 1e-12.
  `nix flake check` runs it (the flake pins the C source as an input);
  locally, `TYPE_NN_SRC=/path/to/type-nn pytest`.
- **The board.** `benchmarks/bench.py` ports `bench.c` exactly: the same
  files, the same xorshift split and shuffle (checked row for row), per-sample
  Adam at `lr x 0.1`, the same epochs and learning rates, and 5 seeds per
  cell. `ref-type-nn` and `ref-type-nn-overfit` are exact configurations of
  C's two models; `benchmarks/board.py` compares each with the C board
  (`benchmarks/reference/`, deterministic, reproduced digit for digit) and
  writes every table of [BOARD.md](https://github.com/hadilq/torch-type-nn/blob/main/BOARD.md) from the result files.

```sh
nix run .#board-all      # every result on all cores, then BOARD.md's tables
```

## Development

Everything is wired through `flake.nix`:

```sh
nix develop            # python + torch + pytest + build tools (or: direnv allow);
                       # also copies the pinned datasets to benchmarks/data
pytest                 # run the tests from the checkout
nix run .#test         # same, without entering the shell
nix flake check        # build the package, run pytest inside the build (with the
                       # pinned datasets), ruff
nix build              # ./result: the installed package
nix run .#dist         # sdist + wheel in ./dist, checked by twine
nix run .#publish -- --repository testpypi   # upload (TestPyPI first)
```

The benchmark datasets are pinned (URL + SRI hash) in
`benchmarks/datasets.json` and never committed; see
[benchmarks/data/README.md](https://github.com/hadilq/torch-type-nn/blob/main/benchmarks/data/README.md).

Without Nix: `pip install -e ".[dev]" && python benchmarks/fetch_data.py && pytest`.

### On a GPU

Every test that takes the `device` fixture also runs on `cuda` when a CUDA
device is visible; `TNN_REQUIRE_CUDA=1` makes a missing GPU an error rather
than a skip. One script runs the GPU suite (the whole test-suite with
`TNN_REQUIRE_CUDA=1`, then the board's training loop for every model on the
device), in two ways:

```sh
nix run .#test-cuda              # on the host, now: no Nix configuration needed
nix build .#cuda-tests -L        # in the build sandbox: needs the `cuda` feature
nix flake check --impure         # adds checks.cuda when /dev/nvidiactl exists
TORCH_TYPE_NN_CUDA=1 nix flake check --impure   # force it (=0: leave it out)
nix develop .#cuda               # torch-bin (CUDA 13.0) shell
nix run .#bench-cuda -- all all --seeds 5 --device cuda
```

A plain `nix flake check` is a pure evaluation, which cannot see the host, so
it never adds the GPU check. With `--impure` it adds `checks.cuda` when a GPU
is visible (`/dev/nvidiactl`) *and* the Nix daemon enables the `cuda` system
feature (read from `/etc/nix/nix.conf`, `/etc/nix/nix.custom.conf`,
`NIX_CONFIG`); a GPU without the feature skips the check with a warning.
`TORCH_TYPE_NN_CUDA=1` / `=0` forces it on / off.

The sandboxed check declares `requiredSystemFeatures = [ "cuda" ]`, so the
daemon must advertise the feature and expose the GPU to the build:

**NixOS:**

```nix
programs.nix-required-mounts = {
  enable = true;
  presets.nvidia-gpu.enable = true;   # adds the cuda/gpu/opengl features and
};                                    # mounts the driver + /dev/nvidia* for them
# Setting nix.settings.system-features replaces the default list, so keep the
# features you rely on (nixos-test, benchmark, big-parallel, kvm, ...):
nix.settings.system-features = [ "nixos-test" "benchmark" "big-parallel" "kvm" ];
```

**Other Linux (Nix daemon):** find the directory holding the driver's
`libcuda.so.1` (`ldconfig -p | grep libcuda.so.1`; e.g. `/usr/lib/x86_64-linux-gnu`
on Debian/Ubuntu, `/usr/lib` on Arch), then in `/etc/nix/nix.conf`:

```
extra-system-features = cuda
extra-sandbox-paths = /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools? /run/opengl-driver/lib=/usr/lib/x86_64-linux-gnu
```

and restart the daemon (`sudo systemctl restart nix-daemon`). torch-bin looks
for the driver in `/run/opengl-driver/lib`; the `target=source` form mounts the
host's driver directory there inside the sandbox only (a trailing `?` marks a
path that may be missing). With several GPUs add `/dev/nvidia1`, ... Check with
`nix build .#cuda-tests -L`.

## Publishing

It's in development state, but you can use download it from PyPI [here](https://pypi.org/project/torch-type-nn/).

## License

MIT. Derived from type-nn by Hadi Lashkati Ghouchani (MIT).
