"""Graph (slime-mould) variant of type-nn.

Nodes replace layers. The graph begins as a small-world digraph and is
grown and pruned by the same dummy rule that the chain TypeNN uses, guided
by inference (positive) and back-propagation (negative) feedbacks.

This module is the starting point for the design described in
ARCHITECTURE.md and the tasks in PLAN.md. The forward pass on a static
graph is functional; dynamic scaling is still being added.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from .layer import AndOr
from .network import birth_depth

__all__ = ["Node", "GraphTypeNN", "GraphScaler"]


class Node(AndOr):
    """A type-nn node: an AndOr that also records its graph neighbours.

    The numerical behaviour is identical to :class:`AndOr`. The extra
    attributes exist so that a graph can route several sources into one
    node (merging) and one node into several targets (branching).
    """

    def __init__(self, *args, node_id: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node_id = int(node_id)
        self.incoming: list[int] = []   # source node ids, in column-block order
        self.outgoing: list[int] = []   # target node ids
        # running feedback statistics (reset by the scaler)
        self.inference_mag = 0.0
        self.backprop_mag = 0.0
        self.edge_inference: dict[int, float] = {}
        self.edge_backprop: dict[int, float] = {}
        # input column ranges owned by each source: src_id -> (start, end)
        self.input_blocks: dict[int, tuple[int, int]] = {}


def _watts_strogatz(n: int, k: int, p: float, generator: torch.Generator) -> list[tuple[int, int]]:
    """Directed Watts–Strogatz: ring lattice plus random rewires.

    Returns a list of (source, target) edges that is guaranteed acyclic
    when nodes are ordered 0 … n-1 (edges only go forward).
    Every node except the first is guaranteed at least one incoming edge.
    """
    if n <= 1:
        return []
    edges = set()
    # local ring (forward only to keep the order topological)
    half = max(1, k // 2)
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if j > i:
                edges.add((i, j))
    # ensure connectivity: every node > 0 has an incoming from the previous
    for i in range(1, n):
        edges.add((i - 1, i))
    # rewires (only non-mandatory edges)
    mandatory = {(i - 1, i) for i in range(1, n)}
    for src, dst in list(edges - mandatory):
        if torch.rand((), generator=generator).item() < p:
            candidates = [t for t in range(src + 1, n) if (src, t) not in edges]
            if candidates:
                new_dst = candidates[int(torch.randint(len(candidates), (), generator=generator).item())]
                edges.discard((src, dst))
                edges.add((src, new_dst))
    return sorted(edges)


class GraphTypeNN(nn.Module):
    """Small-world graph of type-nn nodes.

    At birth the number of nodes is the usual
    :func:`~torch_type_nn.network.birth_depth`. Edges form a directed
    Watts–Strogatz graph (forward only, with a backbone chain so every
    node is reachable). Each node is an :class:`Node` (i.e. an AndOr).

    Multi-source merging concatenates the source outputs; each source owns
    a contiguous block of input columns. Dummy edges (future) will be born
    with zero weights in their block so the function is unchanged.

    Dynamic growth and pruning according to the slime-mould feedbacks are
    the responsibility of a future GraphScaler (see PLAN.md). Until that
    scaler exists the graph is static after construction.

    Args:
        in_features, out_features: task widths (fixed).
        depth: birth depth; default birth_depth(n, m).
        k: mean degree of the initial small-world graph.
        p: rewiring probability.
        seed: RNG seed (same convention as TypeNN).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        depth: int | None = None,
        k: int = 2,
        p: float = 0.1,
        seed: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.generator = torch.Generator().manual_seed(int(seed))
        D = birth_depth(in_features, out_features) if depth is None else int(depth)
        self.birth_depth = D

        # edges first so we know the fan-in of every node
        raw_edges = _watts_strogatz(D, k, p, self.generator)
        self.edges: list[tuple[int, int]] = raw_edges
        incoming: dict[int, list[int]] = defaultdict(list)
        outgoing: dict[int, list[int]] = defaultdict(list)
        for src, dst in raw_edges:
            outgoing[src].append(dst)
            incoming[dst].append(src)

        self.nodes = nn.ModuleList()
        for i in range(D):
            if i == 0:
                n_in = self.in_features
                sources: list[int] = []          # external input, no graph source
            else:
                sources = sorted(incoming.get(i, []))
                # each source contributes out_features columns
                n_in = max(1, len(sources)) * self.out_features
            node = Node(
                n_in, self.out_features, ors_per_unit=1,
                node_id=i, generator=self.generator,
                device=device, dtype=dtype,
            )
            node.incoming = sources
            node.outgoing = sorted(outgoing.get(i, []))
            # record column blocks (only for graph sources)
            start = 0
            for src in sources:
                end = start + self.out_features
                node.input_blocks[src] = (start, end)
                start = end
            self.nodes.append(node)

        # topological order is simply 0 … D-1 because all edges are forward
        self.order = list(range(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the graph.

        A node with several incoming edges receives the concatenation of
        its sources' outputs, in the order stored in ``node.incoming``.
        The first node always receives the external input ``x``.
        """
        acts: dict[int, torch.Tensor] = {}
        for i in self.order:
            node = self.nodes[i]
            if i == 0:
                inp = x
            else:
                parts = [acts[src] for src in node.incoming]
                if not parts:
                    # should not happen after the backbone guarantee
                    inp = torch.zeros(x.shape[0], node.in_features,
                                      device=x.device, dtype=x.dtype)
                else:
                    inp = torch.cat(parts, dim=-1)
            z = node(inp)
            acts[i] = z
            # inference feedback (positive)
            node.inference_mag = float(inp.detach().abs().mean().item())
            if i > 0:
                for src in node.incoming:
                    block = acts[src]
                    node.edge_inference[src] = float(block.detach().abs().mean().item())

            # back-prop feedback (negative) via hook on the output
            def _make_hook(n=node):
                def hook(grad):
                    if grad is not None:
                        n.backprop_mag = float(grad.detach().abs().mean().item())
                return hook
            if z.requires_grad:
                z.register_hook(_make_hook())

            # per-incoming-edge backprop (gradient arriving at this node from each source block)
            # recorded on the source as outgoing feedback
            if i > 0 and inp.requires_grad:
                def _make_in_hook(n=node, sources=list(node.incoming)):
                    def hook(grad):
                        if grad is None:
                            return
                        # grad has shape (B, total_in)
                        offset = 0
                        for src in sources:
                            width = self.out_features
                            block_grad = grad[..., offset:offset + width]
                            # record on the source node as feedback toward this target
                            self.nodes[src].edge_backprop[n.node_id] = float(
                                block_grad.detach().abs().mean().item())
                            offset += width
                    return hook
                inp.register_hook(_make_in_hook())
        return acts[self.order[-1]]


    def add_dummy_edge(self, src: int, dst: int) -> None:
        """Add a forward edge that does not change the function.

        The target node grows ``out_features`` new input columns (all weights
        exactly zero). The source's output is concatenated into those columns
        on the next forward pass, but because the weights are zero the
        numerical result is identical (up to the usual floating-point noise).
        """
        # topological order is maintained in self.order; ids need not be increasing
        if dst == src:
            raise ValueError("self-loops are not allowed")
        if (src, dst) in self.edges:
            return
        target = self.nodes[dst]
        # grow the input block
        for _ in range(self.out_features):
            target.add_input()
        start = target.in_features - self.out_features
        end = target.in_features
        target.incoming.append(src)
        target.input_blocks[src] = (start, end)
        self.nodes[src].outgoing.append(dst)
        self.edges.append((src, dst))
        self.edges.sort()

    def reset_feedback(self) -> None:
        """Clear inference and back-prop magnitude accumulators."""
        for node in self.nodes:
            node.inference_mag = 0.0
            node.backprop_mag = 0.0
            node.edge_inference = {}
            node.edge_backprop = {}

    @property
    def depth(self) -> int:
        """Number of nodes (analogue of layer depth for the board)."""
        return len(self.nodes)

    def num_params(self) -> int:
        return sum(node.num_params() for node in self.nodes)

    def structure(self) -> dict:
        """Diagnostic view of the current graph."""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "live_ors": [node.num_ors().tolist() for node in self.nodes],
            "params": self.num_params(),
            "fan_in": [len(node.incoming) for node in self.nodes],
        }

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"nodes={len(self.nodes)}, edges={len(self.edges)}, "
                f"params={self.num_params()}")



    def insert_node_on_edge(self, src: int, dst: int) -> int:
        """Insert a new identity-ish node between src and dst (forward only).

        The new node is born with a single Or and zero-ish weights so the
        immediate change is small. Full identity preservation and Edit
        reporting are still incomplete (see PLAN.md).
        Returns the new node id.
        """
        if dst <= src:
            raise ValueError("only forward insertion")
        new_id = len(self.nodes)
        # new node reads the source output
        node = Node(self.out_features, self.out_features, ors_per_unit=1,
                    node_id=new_id, generator=self.generator,
                    device=self.nodes[0].weight.device,
                    dtype=self.nodes[0].weight.dtype)
        node.incoming = [src]
        node.input_blocks[src] = (0, self.out_features)
        self.nodes.append(node)
        # rewire: remove direct edge, add src->new and new->dst
        if (src, dst) in self.edges:
            self.edges.remove((src, dst))
            self.nodes[src].outgoing = [t for t in self.nodes[src].outgoing if t != dst]
            self.nodes[dst].incoming = [s for s in self.nodes[dst].incoming if s != src]
            if src in self.nodes[dst].input_blocks:
                del self.nodes[dst].input_blocks[src]
        self.add_dummy_edge(src, new_id)
        self.add_dummy_edge(new_id, dst)
        # insert into topological order between src and dst
        try:
            pos_src = self.order.index(src)
            pos_dst = self.order.index(dst)
            insert_at = pos_src + 1
            if insert_at > pos_dst:
                insert_at = pos_dst
            self.order.insert(insert_at, new_id)
        except ValueError:
            self.order.append(new_id)
        return new_id

class GraphScaler:
    """Early growth / late pruning of edges for a :class:`GraphTypeNN`.

    Mirrors the three-phase schedule of :class:`~torch_type_nn.scaling.StructureScaler`
    (grow on u < 1/3 while the epoch MSE exceeds one sample's share of target
    variance, fit in the middle, prune on u >= 2/3). Decisions use the dummy
    rule and the recorded inference / back-prop feedbacks.

    This is a first implementation of task 5 in PLAN.md. It proposes dummy
    edges in the direction of strong inference signals and drops edges whose
    weight blocks have returned inside the identity band. Node insertion and
    full BIC support are left for later increments.
    """

    def __init__(self, model: GraphTypeNN, optimizer: torch.optim.Optimizer, *,
                 epochs: int, steps_per_epoch: int, rule: str = "threshold") -> None:
        self.model = model
        self.optimizer = optimizer
        for node in model.nodes:
            node.edit_sink = []
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.total = self.epochs * self.steps_per_epoch
        self.step = 0
        self.phase = 0          # GROW
        self.rule = rule
        self.epoch_loss = 0.0
        self.n_train = 0
        self.var_t = 1.0
        from types import SimpleNamespace
        self.counters = SimpleNamespace(edge_add=0, edge_drop=0, node_add=0)

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def begin(self) -> None:
        self.model.reset_feedback()

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> None:
        # y is the model output; we only need the loss contribution for the gate
        self.epoch_loss += float(((y - t).detach() ** 2).sum().item())
        self.n_train += t.shape[0]
        self.step += 1

    def epoch_end(self) -> None:
        if self.n_train == 0:
            return
        mse = self.epoch_loss / self.n_train
        u = self.step / max(1, self.total)
        from .scaling import phase_at
        self.phase = phase_at(u)
        unexplained = mse > (self.var_t / max(1, self.n_train))
        if self.phase == 0 and unexplained:          # GROW
            self._propose_edges()
        elif self.phase == 2:                        # PRUNE
            self._prune_edges()
        # report structural edits so stock optimizers stay in sync
        from .edits import follow_structure
        edits = []
        for node in self.model.nodes:
            if hasattr(node, "edit_sink") and node.edit_sink is not None:
                edits.extend(node.edit_sink)
                node.edit_sink.clear()
        if edits:
            follow_structure(self.optimizer, self.model, edits)
        self.model.reset_feedback()
        self.epoch_loss = 0.0
        self.n_train = 0

    def end(self) -> None:
        self.epoch_end()

    def _propose_edges(self) -> None:
        """Add a few dummy edges toward nodes with high inference magnitude."""
        nodes = self.model.nodes
        if len(nodes) < 2:
            return
        # rank nodes by inference signal
        ranked = sorted(range(len(nodes)), key=lambda i: nodes[i].inference_mag, reverse=True)
        added = 0
        for src in ranked[:2]:                      # strongest sources
            for dst in ranked[-2:]:                  # weakest / later targets
                if dst > src and (src, dst) not in self.model.edges and added < 2:
                    self.model.add_dummy_edge(src, dst)
                    self.counters.edge_add += 1
                    added += 1
                    if added >= 1:
                        # occasionally deepen by inserting a node on a strong edge
                        if self.model.nodes[src].inference_mag > 0.5 and len(self.model.nodes) < 8:
                            self.model.insert_node_on_edge(src, dst)
                            self.counters.node_add += 1
                        return

    def _prune_edges(self) -> None:
        """Drop edges whose weight block is near the identity (all near zero)."""
        band = self.lr * (self.steps_per_epoch ** 0.75)
        to_drop = []
        for src, dst in list(self.model.edges):
            if dst == src + 1 and len(self.model.nodes[dst].incoming) == 1:
                continue                             # keep the backbone for now
            target = self.model.nodes[dst]
            if src not in target.input_blocks:
                continue
            start, end = target.input_blocks[src]
            # mean absolute weight on that block
            w = target.weight[..., start:end]
            if float(w.detach().abs().mean()) < band:
                to_drop.append((src, dst))
        for src, dst in to_drop:
            self._drop_edge(src, dst)
            self.counters.edge_drop += 1

    def _drop_edge(self, src: int, dst: int) -> None:
        """Remove an edge and its input columns (function-preserving if weights ~ 0)."""
        target = self.model.nodes[dst]
        if src not in target.input_blocks:
            return
        start, end = target.input_blocks[src]
        # drop columns from the end so indices stay valid; simpler to rebuild blocks
        # For the first version we only drop if it is the last block
        if end != target.in_features:
            return                                 # conservative: only drop trailing blocks
        for _ in range(end - start):
            target.drop_input(target.in_features - 1)
        target.incoming = [s for s in target.incoming if s != src]
        del target.input_blocks[src]
        # rebuild remaining blocks
        offset = 0
        for s in target.incoming:
            target.input_blocks[s] = (offset, offset + self.model.out_features)
            offset += self.model.out_features
        self.model.nodes[src].outgoing = [t for t in self.model.nodes[src].outgoing if t != dst]
        self.model.edges = [(s, d) for s, d in self.model.edges if not (s == src and d == dst)]
