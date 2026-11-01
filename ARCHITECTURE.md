# Type-NN Architecture

Type-NN is a neural architecture in which every node (or layer) computes a
partition function over an And of Ors, with the natural logarithm as the
activation. The design is taken from the assembly index of assembly theory:
each node counts structured ways of composing its inputs. The source of
truth for the architecture is the description in the repository README of
the original C implementation and the posts
[Type Mechanics](https://hadilq.com/posts/type-mechanics/) and
[Train the Knowledge](https://hadilq.com/posts/train-the-knowledge/).

## The node (partition function)

For an input vector \(x\), unit \(k\) of a node is

\[
\begin{align*}
\mathrm{Or}_{k,r} &= w_{k,r}\cdot x + b_{k,r},\\
A_k &= \prod_r (\mathrm{Or}_{k,r})^{a_{k,r}},\\
z_k &= \operatorname{sign}(A_k)\log(1 + |A_k|),
\end{align*}
\]

where the assembly indices satisfy \(a_{k,r}\ge 1\). In log-space the
product never overflows. Free slots are filled with the identity Or
(\(w=0\), \(b=1\), \(a=1\)), which contributes a factor of exactly 1 and
therefore never changes the output. The network is as dense as an MLP:
every live connection is a full matrix of Ors.

All structural growth and pruning is driven by back-propagation. Scaling
never stores training pairs except when the BIC rule explicitly re-measures
them.

## The three scaling problems and the dummy rule

There are three axes, each governed by the same dummy rule.

- **Or / width.** The output of the previous node (the input of the current
  node) grows a new coordinate. That coordinate is paired with a dummy
  weight born at zero in every Or that reads it. If the dummy weight is
  moved by back-propagation beyond a noise threshold it is promoted; if it
  later returns inside the identity band it is dropped.
- **And / degree.** Every And keeps a dummy noisy identity Or
  (\(w\sim 0\), \(b\sim 1\)). If that Or acquires weight it is promoted and
  a fresh dummy identity Or is inserted. When two dummy identity Ors remain
  after the threshold test, one is dropped (the And keeps the most-moved
  Or).
- **Depth / nodes.** The network is born with
  \(\operatorname{round}(\log(1+n\cdot m))\) nodes, where \(n\) is the input
  size and \(m\) the output size. A new identity node may be inserted
  between *any* two existing nodes (not only at the ends). An identity node
  that stays inside its threshold band is removed.

Probes are exact identities up to a small noise. Promotion occurs when the
root-mean-square displacement from identity exceeds
\(\theta(T)=\eta T^{3/4}\) (\(T\) is the age of the probe). Growth is
allowed only in the early part of training (first third of the schedule)
and only while the training MSE still exceeds one sample’s share of the
target variance. Pruning occurs only in the late part of training (last
third). The middle third is pure fitting. This schedule is part of the
architecture, not a hyper-parameter.

Two decision rules exist:

- **threshold** (default, “type-nn-overfit”): every decision is read from
  the movement of the dummy weights alone.
- **bic**: a moved probe is kept only if it also pays the Bayesian
  information price; pruning ablated items while the criterion
  \(n\ln\mathrm{MSE}+K\ln n\) does not increase.

## Linked-list variant (the reference)

In the ordinary TypeNN the nodes form a linked list (a chain). Each node’s
output is the next node’s input. Depth probes sit in the gaps of this
chain. The implementation in `network.py` / `layer.py` realises exactly
this structure. All three axes are exercised by `StructureScaler` through
the `TypeNNAdapter`.

## Graph variant (slime-mould)

When the model is a graph the layers are called **nodes**. Every node is the
same partition function; the linked list of layers becomes a DAG in
topological order, and the dummy rule runs on the graph. The reference is
type-nn's `type_nn_graph.c` (see type-nn's ARCHITECTURE.md §5); this package
ports it exactly.

- **Reading x.** The first layer of nodes reads the model input x and nothing
  else; every other node reads the outputs of other nodes. The chain is the
  graph whose topology is a linked list.
- **Birth.** `round(ln(1 + n m))` nodes as wide as the chain's layers
  (`n -> m -> ... -> m`, the chain's init law), the backbone `x -> v0 -> ... ->
  v_{D-1}`, plus Watts–Strogatz shortcuts `i -> i+h` (`h = 2 … ceil(ln D)`,
  rewired with probability `1/D`) born as dummies (zero columns). At birth the
  graph computes exactly the chain's function.
- **Edges.** An edge carries some source coordinates to some destination
  columns through an explicit map `k -> j`, and multiplies the value by its
  slime-mould weight `c = in_w · out_w`. `in_w` is the edge's share of the
  destination's accumulated inference feedback (mean |value entering|, per
  x); `out_w` is its share of the source's accumulated back-prop feedback
  (mean |dL/dz| at its destination, per y). In- and out-weights each sum to 1
  per node. When they are re-estimated the destination absorbs the change
  (`w' = w c / c'`) and Adam steps a wrapped column by `lr / c`, so the weight
  is a reparameterisation of the function and of the optimiser; its job is
  to rank the edges for the late drops (thinnest first).
- **Merging** extends the Or list: a merge dummy `u -> v` gives every Or of v
  zero dummy weights for u's outputs. **Branching** emerges: an Or dummy (a
  new output of a node) is read by every out-edge, and late pruning removes
  each (edge, output) column its destination does not pay for.
- **One dummy rule, four axes.** And (a noisy identity Or per unit), Or (a
  trained new output read through zero columns), merge (a dummy edge, one per
  destination, from the source with the largest inference feedback) and depth
  (an identity node on the edge with the largest junction gradient, the
  consumer folded on the epoch's training inputs). A dummy is promoted when it
  moved past `theta(T) = lr T^(3/4)` and pays its BIC price on the training
  data; late, edges are dropped thinnest first and a joint ablation pool removes
  Ors, coordinates and single columns while `n ln MSE + K ln n` does not rise.
  The graph has one rule (BIC); the chain's threshold rule is not ported to it.

### Three implementations, one rule

| class | node storage | training step | role |
|---|---|---|---|
| `GraphTypeNN` + `GraphScaler` | `AndOr` (padded, autograd) | any loop with `TypeAdam` | the user-facing module |
| `NativeGraphTypeNN` | C `TnnLayer` | C (`tnn_py_g_*`) | the reference |
| `CudaGraphTypeNN` | packed `(R, n_in)` per node on a torch device | hand-written per-sample backward + per-Or Adam | the device backend |

The rule (`torch_type_nn/graph/core.py`) is written once against a small
`NodeStore` interface; the backends differ only in how a node stores its Ors.
Every random draw is C's `xorshift32` stream in C's order, so all three are
born with the C graph's weights, and take the same per-sample step to 1e-12.
Started from the C graph's state at any epoch, they take that epoch's
structural decisions. A long run drifts from C's path, because per-sample
training amplifies rounding (tests/test_graph.py,
`benchmarks/graph_parity.py`).

## Faithfulness

An implementation is faithful when:

1. every node realises the partition-function formula above,
2. every structural edit is an identity up to the declared noise (so the
   function is unchanged until back-propagation moves the probe),
3. promotion and pruning thresholds, the three-phase schedule, and the
   initial depth formula are respected,
4. benchmarks use identical data splits, standardisation, epoch budgets,
   learning rates and loss for every model, and report structure as well as
   accuracy,
5. any deliberate difference from the description above is documented.

The chain implementation satisfies these criteria (see `docs/AUDIT.md` and
`docs/DIFFERENCES.md`). The graph variant is held to the same standard by
checking it against the C graph number for number: births, single steps and
single epochs from the same state (`tests/test_graph.py`). Whole runs are
compared over seeds on the board.
