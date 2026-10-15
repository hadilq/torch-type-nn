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

When the model is a graph the same objects are called **nodes** rather than
layers. The dummy rule is applied to the edges and nodes of a directed
graph instead of to the links of a chain.

- The graph is initialised as a small-world network (high local clustering,
  short characteristic path length).
- Early in training new edges (and, when useful, new nodes) are created;
  late in training edges and nodes that have returned to identity are
  dropped.
- Growth is guided by two feedbacks inspired by the
  [Slime Mould Algorithm](https://en.wikiversity.org/wiki/Slime_Mould_Algorithm):
  - **Inference feedback** (positive). For each input \(x\) one records how
    large the inputs arriving at each node are. The structure grows in the
    direction of this signal.
  - **Back-propagation feedback** (negative). For each output \(y\) one
    records the gradient magnitudes. This feedback inhibits or prunes.
- A node may be both a **branching** node and a **merging** node.
  - Merging: the Or list of the node simply contains additional weight
    columns (or additional dummy Ors) for every incoming edge. The dummy
    rule of Or is extended so that each potential source node may contribute
    its own set of dummy weights.
  - Branching: distinct coordinates (or distinct groups of coordinates) of
    the node’s output are routed to distinct target nodes.
- All scaling still occurs inside the back-propagation calculation; the
  forward pass remains a pure evaluation of the partition functions on the
  current graph.

The graph variant therefore preserves the mathematical definition of every
node, the dummy-rule semantics, the early-grow / late-prune schedule, and
the density of connections, while replacing the linked-list topology by a
dynamically evolving small-world graph.

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

The existing chain implementation satisfies these criteria (see
`docs/AUDIT.md` and `docs/DIFFERENCES.md`). The graph variant must be held
to the same standard.
