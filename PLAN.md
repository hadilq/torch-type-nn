# Plan: Graph (Slime-Mould) Variant of Type-NN

The reference TypeNN is a linked list of AndOr layers. The graph variant
replaces that list by a directed small-world graph of nodes while keeping
every other rule of the architecture (partition-function nodes, dummy
identities, three-phase schedule, early growth / late pruning). The design
is specified in `ARCHITECTURE.md`.

The tasks below are ordered so that each step remains testable and the
benchmark harness stays fair.

## 1. Node abstraction

- [x] Rename the conceptual object from “layer” to “node” in the graph
      variant (the existing `AndOr` class may be subclassed or reused).
- [x] Give each node a stable identity, a set of incoming edges and a set
      of outgoing edges.
- [ ] Ensure a node with a single incoming edge and a single outgoing edge
      behaves identically to the corresponding `AndOr` layer (bit-level
      agreement on the identity case).

## 2. Initial small-world graph

- [x] Construct a Watts–Strogatz (or equivalent) digraph whose number of
      nodes is the usual birth depth \(\operatorname{round}(\log(1+n m))\).
- [x] Guarantee a topological order so that the forward pass is well-defined.
- [ ] Seed the generator exactly as `TypeNN` does, so the birth draw is
      reproducible.
- [ ] Unit test: the initial graph is small-world (high clustering, short
      paths) and every node starts with a single live Or.

## 3. Multi-input (merging) and multi-output (branching)

- [x] Merging node: the input vector of a node is the concatenation of the
      outputs of its sources. Each source owns its own block of weight
      columns; the dummy-Or rule is extended so that a new incoming edge
      is born with dummy weights (exactly zero) for every coordinate of
      that source.
- [x] Branching node: distinct output coordinates (or groups of them) may
      be routed to distinct targets. A new outgoing edge is a dummy that
      does not change the function until back-propagation moves it. (dummy-edge insertion implemented; full coordinate splitting later)
- [ ] A node may be both merging and branching.
- [x] Identity property: adding a dummy edge never changes the numerical
      output (within floating-point noise of the declared amplitude).

## 4. Feedback signals

- [x] Inference feedback (positive): during each forward pass record, per
      node and per incoming edge, a running statistic of the magnitude of
      the arriving signal (e.g. mean absolute input or mean absolute
      pre-activation).
- [x] Back-propagation feedback (negative): during each backward pass
      record, per node and per outgoing edge, a running statistic of the
      gradient magnitude.
- [x] Both statistics are reset at epoch boundaries and are visible to the
      scaler; they never store raw training pairs. (reset_feedback provided; scaler integration next)

## 5. Graph-aware dummy rule and scaler

- [x] Early phase (\(u<1/3\)): propose new edges (and, when the inference
      feedback is strong enough, new nodes) in the direction of the
      inference signal, subject to the same MSE-gate used by the chain
      scaler. (GraphScaler proposes dummy edges and can insert nodes on strong edges)
- [x] Late phase (\(u\ge 2/3\)): drop edges and nodes that have returned
      inside the identity band, exactly as the threshold (or BIC) rule
      does for Ors, Ands and layers today. (edge dropping implemented for trailing blocks)
- [ ] Depth probes become node probes that may be inserted on any edge
      (or between any pair of nodes that preserves acyclicity).
- [ ] Re-use `StructureScaler` where possible; if a separate
      `GraphScaler` is required, it must implement the identical schedule,
      thresholds and decision rules.
- [ ] All structural edits are reported as `Edit` objects so that both
      `TypeAdam` and stock optimisers continue to work.

## 6. Faithfulness tests

- [ ] Identity probes do not change the forward function.
- [ ] A probe is promoted only when its displacement exceeds \(\theta(T)\).
- [ ] The three-phase schedule is respected (no growth after \(u=1/3\),
      no pruning before \(u=2/3\)).
- [ ] Merging and branching preserve the partition-function formula.
- [ ] Numerical agreement with a pure chain TypeNN on a graph that has
      been forced into a chain.
- [ ] Gradient check (finite differences) on a small static graph.

## 7. Benchmark fairness

- [x] Add a `graph-type-nn` (and optionally `graph-type-nn-bic`) entry to
      `benchmarks/bench.py` and `benchmarks/scale.py`. (basic entry added; uses GraphScaler)
- [ ] Use exactly the same data files, 70/30 xorshift split, train-only
      standardisation, epoch budgets, learning rates, loss and seed list
      as every other model.
- [ ] Report, in addition to MSE / accuracy, the final number of nodes,
      edges, live Ors and parameters.
- [ ] Do not tune any hyper-parameter of the graph variant on the hold-out
      sets; any schedule or threshold change must be justified by the
      architecture description alone and documented in `docs/DIFFERENCES.md`.
- [ ] Archive pre-audit results if the harness changes.

## 8. Documentation and packaging

- [ ] Update `README.md` with a short “Graph variant” section that points
      to `ARCHITECTURE.md`.
- [ ] Export the new classes from `torch_type_nn/__init__.py`.
- [ ] Add a note in `docs/ADAPTERS.md` describing the graph adapter.
- [ ] Keep `VERSIONS.md` and the changelog current.

## 9. Optional later work

- [ ] CUDA / native back-ends for the graph variant (after the pure-PyTorch
      version is validated).
- [ ] Visualisation of the evolving small-world graph.
- [ ] Comparison against other dynamic-graph architectures under the same
      fair benchmark protocol.

Completion of tasks 1–7 is sufficient for a first faithful release of the
graph variant.
