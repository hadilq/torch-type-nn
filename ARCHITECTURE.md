# torch-type-nn Architecture

torch-type-nn is a PyTorch port of the type-nn architecture. Each computational unit evaluates a partition function derived from type-generating functions: an **Or** is a sum-type (dense linear combination), an **And** is a product-type, and the assembly index comes from Assembly Theory. The activation is the natural logarithm, turning multiplicative counts into additive information (nats).

The network is dense in the same sense as an MLP: the output of one unit is the input of the next. Width, degree (number of factors in an And), and depth are not hyperparameters. They are the three scaling problems solved during training by the **dummy rule**.

## Core Computation

Each layer (or, in the graph extension, each node) maps a collection of input coordinates to a vector of outputs. For output coordinate \(k\):

\[
\text{Or}_{k,r} = w_{k,r} \cdot x + b_{k,r}
\]

\[
A_k = \prod_r \text{Or}_{k,r}^{a_{k,r}}
\]

\[
z_k = \operatorname{sign}(A_k) \ln(1 + |A_k|)
\]

- \(x\) is the concatenation of the outputs of the predecessor units that feed this unit.
- \(a_{k,r} \ge 1\) is the learned assembly index. Exponentiation preserves sign: \(o^a := \operatorname{sign}(o) |o|^a\).
- The forward pass is performed in log space to avoid overflow.

All scaling decisions occur at epoch boundaries from quantities produced by back-propagation on the training set. Gradients are computed with pre-update weights; each Or then takes an Adam step and the assembly index is projected back to \(a \ge 1\).

Two decision rules are supported:

- **threshold** (default, matches C `type-nn-overfit`): a probe is promoted purely on displacement past \(\theta(T) = \mathrm{lr} \cdot T^{3/4}\). Late pruning drops structure that has returned inside the identity band.
- **bic** (matches C `type-nn`): promotion and pruning additionally require the Bayesian information criterion \(n \ln(\mathrm{MSE}_\text{without}/\mathrm{MSE}_\text{with}) > k \ln n\), measured by ablating the item to its identity and re-evaluating the epoch’s training pairs.

## The Dummy Rule

All three scaling problems follow one rule. A **probe** is an identity element that back-propagation is free to move. A probe that moves outside a dynamic noise band is **promoted** (and a fresh probe is inserted). Late in training, structure that falls back inside the identity band or fails an evidence criterion is **dropped**.

- **Or scaling (width / alternatives)**: the predecessor grows a new output coordinate; the current unit meets it with a weight column born at zero.
- **And scaling (degree / conditions)**: each And carries a dummy noisy identity Or (\(w \approx 0\), \(b \approx 1\)). Promotion adds a new dummy; two identity Ors after thresholding cause one to be dropped.
- **Depth scaling (abstraction)**: an identity layer/node may be inserted between any two existing units (the gap with the largest mean input-gradient magnitude, or, under the slime-mold rule, the most desirable cut). An identity unit may later be removed.

Growth is gated by residual error and occurs early; pruning occurs late.

## Graph Extension and Slime-Mold Growth

The original depth scaling operated on a linked list of layers. The graph variant generalises this structure to a directed acyclic graph of **nodes**.

- A node may be a **merging node**: it receives inputs from several predecessor nodes (realised by additional Or weight blocks).
- A node may be a **branching node**: its output coordinates are consumed by several successor nodes (realised by assigning different subsets of the node’s outputs).
- A node may be both.

The dummy rule for Or is extended so that a merging node maintains dummy weight columns for each distinct source node from which it may receive input.

In addition, connectivity evolves according to a flow-based growth process inspired by Physarum polycephalum (slime mold) network algorithms. Two flows are tracked:

1. **Inference flow** (positive). The magnitude of the activations arriving at each node / traversing each edge. Structure grows preferentially in the direction of large inference flow.
2. **Back-propagation flow** (negative). The magnitude of the gradients flowing backward. Structure tries to decrease this flow.

A conductivity \(D_e\) for each candidate edge (or dummy column) is updated at epoch boundaries:

\[
D_e \leftarrow D_e + \alpha \bigl( f_{\text{inf}}(e) - \lambda f_{\text{bp}}(e) - D_e \bigr)
\]

High-conductivity dummy edges/nodes are offered to the ordinary dummy-rule (displacement + optional BIC) acceptance test. The slime-mold dynamics therefore only propose candidates; they do not override the evidence criterion.

In the current linear-stack implementation the slime-mold variation is realised by:

- accumulating mean absolute activations (inference flow) and mean absolute input-gradients (back-prop flow) at every junction;
- using the desirability \(f_{\text{inf}} - \lambda f_{\text{bp}}\) to choose the gap for the depth probe and to bias which width probes are most likely to be materialised;
- remaining fully compatible with both the threshold and bic decision rules.

## Distillation Ladder

A longer-term goal is to distill large teacher models into a type-nn (or type-nn-graph) student. The protocol transfers knowledge via logits, hidden-state matching, and optionally structural traces while the student remains free to grow and prune via the dummy rule and slime-mold dynamics. A ladder of increasing capacity starts from tiny open-source models and culminates in distilling a Qwen ~8B-class flash variant (referred to as “Qwen 3.8 flash”) into the graph architecture. See `PLAN.md` for the concrete steps.

## Backends

- Pure PyTorch (`TypeNN` + `StructureScaler`) — the reference.
- Native C wrapper (`NativeTypeNN`) for exact comparison with the original type-nn board.
- CUDA fused backend (`CudaTypeNN`) for larger experiments.

The numerical content of the forward, backward and structural edits matches the C reference to high precision (see `docs/DIFFERENCES.md` and the cross-check tests).
