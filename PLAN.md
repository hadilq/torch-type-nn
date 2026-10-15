# Implementation Plan: Graph-Structured type-nn with Slime-Mold Growth and Distillation Ladder

This plan lists the concrete tasks required to evolve the current linked-list torch-type-nn into a graph of nodes whose connectivity is grown by the dummy rule together with a two-flow slime-mold process, and to build a distillation ladder that ultimately transfers knowledge from a Qwen ~8B-class flash model into the graph architecture.

Tasks are ordered by dependency. The linear-stack slime-mold variation (already started) serves as an intermediate milestone that can be evaluated on the existing board.

## Phase 0 – Baseline and Documentation

- [x] Freeze the current linear implementation as a reference (`TypeNN`, `StructureScaler`, threshold and bic rules).
- [x] Write the high-level design in `ARCHITECTURE.md`.
- [ ] Populate `docs/spec/` (or keep extending the existing docs) with detailed design notes for the graph, slime-mold dynamics and distillation protocol.
- [ ] Add a short section to `README.md` pointing to `ARCHITECTURE.md` and `PLAN.md`.

## Phase 1 – Slime-Mold Variation on the Linear Stack (immediate)

- [x] Track inference-flow (mean |activation|) and back-propagation-flow (mean |∂L/∂x|) statistics at every junction during the training epoch.
- [x] Implement a conductivity / desirability update of the form \(f_{\text{inf}} - \lambda f_{\text{bp}}\) and use it to:
  - select the gap for the depth probe (generalising the current “loudest gap”);
  - (width-probe bias left for later; depth is the primary use)
- [x] Expose the behaviour as a new rule (or a flag) on `StructureScaler` so that both threshold and bic acceptance remain available.
- [x] Add board models `type-nn-slime` and `type-nn-slime-bic` that use the new rule.
- [x] Verify that the path-graph special case (linear stack) reproduces the existing board numbers within noise when the slime-mold bias is disabled. (bench-all green)
- [ ] Unit tests for flow accumulation, desirability ordering, and that function-preserving edits still hold.

## Phase 2 – Data Structures for the Graph

- [x] Introduce a `Node` type that replaces `AndOr` / layer. A node owns its output vector, its list of Ors (via an internal AndOr), and lists of incoming and outgoing edges. (`src/torch_type_nn/graph.py`)
- [x] Represent the network as a DAG: each node stores predecessor and successor edges.
- [ ] Support multiple incoming edges (merging): an Or’s weight vector is the concatenation of blocks, one block per source node (or per source output coordinate). (data structures ready; arithmetic still uses a single concatenated input)
- [ ] Support multiple outgoing edges (branching): each successor receives a (possibly different) subset of the node’s output coordinates. (Edge.coords exists; selection is implemented in the forward gather)
- [x] Provide conversion utilities that turn the classic linear stack into a trivial path graph (for regression testing). (`graph_from_linear`)
- [x] Keep the existing Or / And arithmetic inside a node so that the numerical content remains unchanged. (Node.core is an AndOr)

## Phase 3 – Forward and Backward Passes on the Graph

- [ ] Topological sort (or maintain an explicit topological order) for the forward sweep.
- [ ] Concatenate the outputs of all predecessors when evaluating a node’s Ors.
- [ ] Accumulate gradients into each predecessor’s output buffer according to the edges that were used.
- [ ] Verify gradients with central finite differences on small hand-crafted graphs (single merge, single branch, diamond, etc.).

## Phase 4 – Dummy Rule and Slime-Mold on the Graph

- [ ] **Or / merging probes**: each node maintains a set of dummy weight columns, one for every currently possible source node (or for a candidate set of sources). A dummy column that moves past the threshold (and, under bic, pays for itself) is promoted to a real incoming edge; a real edge whose column falls back inside the identity band is dropped.
- [ ] **And probes**: unchanged in spirit; each And still carries one identity Or. The identity Or’s weight vector simply lives in the concatenated weight space.
- [ ] **Node / depth probes**: an identity node may be inserted on any cut of the DAG (the cut whose slime-mold desirability is highest). Insertion and removal must preserve the near-exact affine correction already used for linear depth edits.
- [ ] Generalise the BIC evidence measurement so that ablating an edge, an Or, or a node is performed by resetting the corresponding parameters to identity and re-evaluating the cached training pairs.
- [ ] During the forward and backward passes, accumulate inference-flow and back-prop-flow statistics for every edge / dummy column.
- [ ] Apply the conductivity update and use it only to propose candidates; the ordinary dummy-rule (displacement + optional BIC) decides acceptance.
- [ ] Ensure the slime-mold update never reads the hold-out set and remains compatible with the existing grow / fit / prune schedule.

## Phase 5 – Integration, Testing, and Benchmarking

- [ ] Unit tests for every graph motif (merge, branch, identity-node insertion/removal, multi-source Or).
- [ ] Gradient checks on the new data structures.
- [ ] Reproduce the existing board numbers with the path-graph special case (regression).
- [ ] Run the full board with the slime-mold graph and compare parameter counts, training time, and hold-out error against the linear baseline, the linear slime variation, and `mlp`.
- [ ] Profile memory and compute; the graph must remain practical for the current small-data regime and for the first rungs of the distillation ladder.

## Phase 6 – Distillation Ladder

- [ ] Define a distillation protocol that transfers knowledge from a teacher model (logits, hidden states, or structural traces) into a type-nn student. The student remains free to grow and prune via the dummy rule and slime-mold dynamics.
- [ ] Implement student-side hooks so that a type-nn (or type-nn-graph) can be trained to match a teacher’s outputs (and optionally intermediate activations).
- [ ] Build a ladder of increasing capacity:
  - Tiny open-source models (or distilled subsets) as first teachers.
  - Progressively larger models (e.g. small Qwen or Llama variants).
  - Final target: distill **Qwen 3.8 flash** (or the closest publicly available Qwen ~8B-class flash / Qwen2.5-3B / Qwen2.5-7B flash-style checkpoint) into a type-nn graph.
- [ ] Measure parameter efficiency, inference cost, and fidelity (KL / task metrics) at each rung of the ladder.
- [ ] Document the protocol and results under `docs/spec/distillation.md` (or equivalent).

## Phase 7 – Optional Refinements

- [ ] Dynamic candidate-source selection (do not keep a dummy column for every existing node; only for nodes that already have high inference flow).
- [ ] Soft topological constraints or cycle detection if recurrent motifs are ever desired.
- [ ] Formal verification (or additional Coq proofs, if the C suite is extended) for the new identity lemmas (adding a dummy multi-source Or or an identity node on a cut does not change the network function).

## Deliverables

- Updated source files implementing the linear slime-mold variation and, later, the full graph.
- Passing tests and an extended board that includes the slime-mold models.
- `ARCHITECTURE.md`, `PLAN.md`, and supporting documentation kept in sync with the code.
- A reproducible distillation protocol and results for the ladder up to the Qwen target.
