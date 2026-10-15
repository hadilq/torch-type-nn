"""graph-type-nn as an ``nn.Module``: :class:`GraphTypeNN` and :class:`GraphScaler`.

Every node is an :class:`~torch_type_nn.layer.AndOr` (the same partition
function, padded ``(m, R, n)`` storage and exact backward as a chain layer);
the forward is batched and differentiable, so any training loop works.
Structure changes at epoch boundaries through :class:`GraphScaler`, which
runs the C rule (:mod:`torch_type_nn.graph.core`).

Optimizer: :class:`~torch_type_nn.optim.TypeAdam`. The rule rescales the
weights of a column when an edge weight is re-estimated (w' = w c / c') or
a depth edit is folded, and moves the Adam moments with them; TypeAdam
keeps those moments in the node, and steps a column wrapped by edge
weight ``c`` by ``lr / c`` (``AndOr.col_lr``), exactly as the C code does.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from ..layer import AndOr
from .core import LR_SCALE, GraphCore, NodeStore

__all__ = ["GraphTypeNN", "GraphScaler", "TorchStore", "Node"]


class Node(AndOr):
    """A graph node: an :class:`AndOr` with a per-column Adam scale."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.col_lr: torch.Tensor | None = None


class TorchStore(NodeStore):
    """An :class:`AndOr` node seen through the rule's interface.

    ``order[k]`` lists the live slots of unit ``k`` in C order (an Or is
    appended when added; the order is kept when one is dropped), because
    the padded layer reuses free slots.
    """

    def __init__(self, layer: Node, order: list[list[int]]) -> None:
        self.L = layer
        self.order = order

    @classmethod
    def empty(cls, n_in: int, n_out: int, device, dtype) -> "TorchStore":
        layer = Node(n_in, n_out, 1, device=device, dtype=dtype)
        with torch.no_grad():
            layer.mask.fill_(False)
            layer.project_()
            layer.reset_adam_state()
        return cls(layer, [[] for _ in range(n_out)])

    # sizes
    @property
    def n_in(self) -> int:
        return self.L.in_features

    @property
    def n_out(self) -> int:
        return self.L.out_features

    def n_or(self, k: int) -> int:
        return len(self.order[k])

    def total_ors(self) -> int:
        return sum(len(o) for o in self.order)

    def num_params(self) -> int:
        return self.total_ors() * (self.n_in + 2)

    # Ors
    def get_or(self, k, r):
        s = self.order[k][r]
        L = self.L
        return (L.weight[k, s].detach().to(torch.float64).cpu().clone(),
                float(L.bias[k, s].detach()), float(L.assembly[k, s].detach()))

    @torch.no_grad()
    def set_or(self, k, r, w=None, b=None, a=None):
        s = self.order[k][r]
        if w is not None:
            self.L.weight[k, s] = w.to(self.L.weight)
        if b is not None:
            self.L.bias[k, s] = b
        if a is not None:
            self.L.assembly[k, s] = a

    def or_probe(self, k, r):
        return bool(self.L.or_probe[k, self.order[k][r]])

    @torch.no_grad()
    def set_or_probe(self, k, r, v):
        self.L.or_probe[k, self.order[k][r]] = bool(v)

    def or_born(self, k, r):
        return int(self.L.or_born[k, self.order[k][r]])

    @torch.no_grad()
    def add_or(self, k, w, b, a, probe, born):
        s = self.L.add_or(k, w.to(self.L.weight), float(b), float(a), probe=bool(probe),
                          born=int(born))
        self.order[k].append(s)

    @torch.no_grad()
    def drop_or(self, k, r):
        s = self.order[k].pop(r)
        self.L.drop_or(k, s)

    # units
    def unit_probe(self, k):
        return bool(self.L.unit_probe[k])

    @torch.no_grad()
    def set_unit_probe(self, k, v):
        self.L.unit_probe[k] = bool(v)

    def unit_born(self, k):
        return int(self.L.unit_born[k])

    @torch.no_grad()
    def set_unit_born(self, k, v):
        self.L.unit_born[k] = int(v)

    def add_unit(self) -> int:
        k = self.L.add_unit()
        self.order.append([])
        return k

    def drop_unit(self, k):
        self.L.drop_unit(k)
        self.order.pop(k)

    # columns
    def add_input(self) -> int:
        self.L.add_input()
        return self.n_in - 1

    def drop_input(self, j):
        self.L.drop_input(j)

    @torch.no_grad()
    def scale_column(self, j, alpha, beta):
        L = self.L
        L.bias.add_(L.weight[:, :, j] * beta)
        L.weight[:, :, j] *= alpha
        L.adam_m_weight[:, :, j] /= alpha
        L.adam_v_weight[:, :, j] /= alpha * alpha
        L.project_()

    def _rows(self):
        ks, ss = [], []
        for k, o in enumerate(self.order):
            for s in o:
                ks.append(k)
                ss.append(s)
        dev = self.L.weight.device
        return torch.tensor(ks, device=dev, dtype=torch.long), \
            torch.tensor(ss, device=dev, dtype=torch.long)

    def cols_get(self, js):
        k, s = self._rows()
        if not len(js) or not len(k):
            return torch.zeros(len(k), len(js), dtype=self.L.weight.dtype)
        w = self.L.weight.detach()[k, s]                    # (ors, n_in)
        return w[:, torch.tensor(js, device=w.device)].clone()

    @torch.no_grad()
    def cols_set(self, js, values):
        k, s = self._rows()
        if not len(js) or not len(k):
            return
        jj = torch.tensor(js, device=self.L.weight.device)
        v = values.to(self.L.weight)
        for c, j in enumerate(jj.tolist()):
            self.L.weight[k, s, j] = v[:, c]

    def forward(self, x):
        return self.L(x)

    def clone(self) -> "TorchStore":
        return TorchStore(copy.deepcopy(self.L), [list(o) for o in self.order])


class _TorchCore(GraphCore):
    """The rule over :class:`TorchStore` nodes, attached to its module."""

    module: "GraphTypeNN | None" = None

    def new_store(self, n_in, n_out):
        return TorchStore.empty(n_in, n_out, self.device, self.dtype)

    def on_structure(self) -> None:
        super().on_structure()
        if self.module is not None and not self._detached:
            self.module._sync_nodes()

    def on_gains(self) -> None:
        super().on_gains()
        if self.module is not None and not self._detached:
            for v, n in enumerate(self.nodes):
                n.store.L.col_lr = self.col_lr(v).to(n.store.L.weight)


class GraphTypeNN(nn.Module):
    """type-nn on a DAG of partition-function nodes (the C ``graph-type-nn``).

    At birth the graph is the chain ``x -> v0 -> ... -> v_{D-1}`` (``D =
    round(ln(1 + n m))``, widths ``n -> m -> ... -> m``, the chain's init law)
    plus small-world shortcuts born as dummies, so it computes exactly the
    chain born from the same seed. Every random draw is C's ``xorshift32``
    stream in C's order: the weights at birth are the C graph's.

    Each edge multiplies the values it carries by its slime-mould weight
    ``c = in_w * out_w`` (inference feedback -> in-weights, back-prop feedback
    -> out-weights, each summing to 1 per node). Train with
    :class:`GraphScaler` (or :func:`fit_graph`) to grow and prune it.

    Args:
        in_features, out_features: task widths.
        seed: the xorshift32 seed (bench.c: ``34972 + 7919 (s + 1)``).
    """

    def __init__(self, in_features: int, out_features: int, *, seed: int = 1,
                 device=None, dtype=torch.float64) -> None:
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.seed = int(seed)
        core = _TorchCore(in_features, out_features, seed, device=device, dtype=dtype)
        self.nodes = nn.ModuleList()
        object.__setattr__(self, "core", core)
        core.module = self
        core.build()
        self._sync_nodes()
        core.on_gains()

    # the board / repr protocol
    @property
    def depth(self) -> int:
        return len(self.core.nodes)

    @property
    def birth_depth(self) -> int:
        return self.core.birth_vertices

    @property
    def n_edges(self) -> int:
        return len(self.core.edges)

    @property
    def edges(self):
        return self.core.edges

    def num_params(self) -> int:
        return self.core.num_params()

    def structure(self) -> dict:
        return self.core.structure()

    def counters(self) -> dict:
        return self.core.counters.as_dict()

    def check(self) -> tuple[bool, str]:
        return self.core.check()

    def _sync_nodes(self) -> None:
        self.nodes = nn.ModuleList([n.store.L for n in self.core.nodes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        train = self.training and torch.is_grad_enabled()
        prev, self.core.training = self.core.training, self.core.training and train
        y = self.core.forward(x2, hooks=self.core.training)
        self.core.training = prev
        return y.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"nodes={self.depth}, edges={self.n_edges}, params={self.num_params()}")

    @classmethod
    def from_snapshot(cls, snap: dict, *, device=None, dtype=torch.float64) -> "GraphTypeNN":
        """A Python graph holding exactly a C graph's state
        (:meth:`~torch_type_nn.native.NativeGraphTypeNN.snapshot`)."""
        m = cls(snap["n_in"], snap["n_out"], seed=1, device=device, dtype=dtype)
        load_snapshot(m.core, snap)
        m._sync_nodes()
        m.core.on_gains()
        return m


def load_snapshot(core: GraphCore, snap: dict) -> None:
    """Replace ``core``'s state by a C snapshot (nodes, Ors with their Adam
    state, edges with weights and feedback, counters, scalars)."""
    from .core import Counters, Edge, NodeState
    core.nodes, core.edges, core.xin = [], [], []
    for n in snap["nodes"]:
        st = core.new_store(n["n_in"], n["n_out"])
        for k, u in enumerate(n["units"]):
            for o in u["ors"]:
                st.add_or(k, o["w"], o["b"], o["a"], o["probe"], o["born"])
            st.set_unit_probe(k, u["probe"])
            st.set_unit_born(k, u["born"])
        _load_adam(st, n)
        fb = n["feedback"]
        core.nodes.append(NodeState(st, probe=n["probe"], born=n["born"], inf_acc=fb[0],
                                    bp_acc=fb[1], ep_inf=fb[2], ep_bp=fb[3],
                                    acc_n=fb[4], ep_n=fb[5]))
        core.xin.append(None)
    for e in snap["edges"]:
        fb = e["feedback"]
        core.edges.append(Edge(e["src"], e["dst"], [list(p) for p in e["pairs"]],
                               in_w=e["in_w"], out_w=e["out_w"], c=e["c"],
                               in_acc=fb[0], out_acc=fb[1], grad_acc=fb[2], ep_in=fb[3],
                               ep_out=fb[4], ep_grad=fb[5], acc_n=fb[6], ep_n=fb[7],
                               probe=e["probe"], born=e["born"]))
    c = snap["counters"]
    core.counters = Counters(**{k: c[k] for k in Counters.__dataclass_fields__})
    sc = snap["scalars"]
    core.lr, core.step, core.total = sc["lr"], sc["step"], sc["total"]
    core.phase, core.crit_best = sc["phase"], sc["crit_best"]
    core.epoch_loss, core.epoch_base = sc["epoch_loss"], sc["epoch_base"]
    core.rng[0] = sc["rng"]
    core.n_train, core.epochs = sc["n_train"], sc["epochs"]
    core.birth_vertices, core.birth_edges = snap["birth_vertices"], snap["birth_edges"]
    core.on_structure()


def _load_adam(st, n: dict) -> None:
    """Adam moments and step counts of every Or, into a store that has them."""
    loader = getattr(st, "load_or_state", None)
    if loader is None:
        return
    for k, u in enumerate(n["units"]):
        for r, o in enumerate(u["ors"]):
            loader(k, r, o)


def _torch_load_or_state(self: TorchStore, k: int, r: int, o: dict) -> None:
    s = self.order[k][r]
    L = self.L
    with torch.no_grad():
        L.adam_m_weight[k, s] = o["mw"].to(L.weight)
        L.adam_v_weight[k, s] = o["vw"].to(L.weight)
        L.adam_m_bias[k, s], L.adam_v_bias[k, s] = o["mb"], o["vb"]
        L.adam_m_assembly[k, s], L.adam_v_assembly[k, s] = o["ma"], o["va"]
        L.adam_step[k, s] = o["t"]


TorchStore.load_or_state = _torch_load_or_state


class GraphScaler:
    """Grows and prunes a :class:`GraphTypeNN` at epoch boundaries (the C rule).

    Drive it from any training loop that steps a
    :class:`~torch_type_nn.optim.TypeAdam` over the model::

        opt = TypeAdam(model, lr=lr)
        scaler = GraphScaler(model, opt, epochs=E, steps_per_epoch=S, n_train=N)
        scaler.begin()
        for epoch in range(E):
            for x, t in loader:
                y = model(x); loss = mse_loss(y, t)
                opt.zero_grad(); loss.backward(); opt.step()
                scaler.observe(x, y, t)
            scaler.epoch_end()
        scaler.end()

    ``lr`` is read from the optimizer (it is the effective step: bench.c
    uses the task lr x 0.1). The schedule counts optimizer steps, so
    ``batch_size=1`` reproduces C's per-sample protocol.
    """

    def __init__(self, model: GraphTypeNN, optimizer: torch.optim.Optimizer, *,
                 epochs: int, steps_per_epoch: int, n_train: int | None = None,
                 rule: str = "bic") -> None:
        from ..optim import TypeAdam
        if not isinstance(optimizer, TypeAdam):
            raise TypeError("GraphScaler needs TypeAdam: the rule rescales columns and "
                            "their Adam moments, which TypeAdam keeps in the node")
        if rule != "bic":
            raise ValueError("graph-type-nn has one rule: the BIC evidence rule "
                             "(type_nn_graph.c); rule='threshold' is chain-only")
        self.model, self.optimizer = model, optimizer
        self.epochs, self.steps_per_epoch = int(epochs), int(steps_per_epoch)
        self.n_train = int(n_train if n_train is not None else steps_per_epoch)
        self.rule = rule

    @property
    def core(self) -> GraphCore:
        return self.model.core

    @property
    def counters(self):
        return self.core.counters

    @property
    def phase(self) -> int:
        return self.core.phase

    @property
    def step(self) -> int:
        return self.core.step

    def begin(self) -> None:
        lr = float(self.optimizer.param_groups[0]["lr"])
        self.core.begin(self.n_train, self.epochs, lr,
                        total=self.steps_per_epoch * self.epochs)
        self.core.training = True

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> None:
        self.core.observe(x, y, t)

    def epoch_end(self) -> None:
        self.core.training = False
        self.core.epoch_end()
        self.core.training = True

    def end(self) -> None:
        self.core.training = False
        self.core.end()


def fit_graph(model: GraphTypeNN, X: torch.Tensor, Y: torch.Tensor, *, epochs: int,
              lr: float, batch_size: int = 1, shuffle=True,
              generator: torch.Generator | None = None) -> GraphScaler:
    """Train ``model`` with TypeAdam at ``lr`` (the effective step) and grow and
    prune it. ``shuffle`` is ``True``, ``False`` or ``fn(epoch, n) -> order``."""
    import math

    from ..optim import TypeAdam
    from ..train import mse_loss
    n = X.shape[0]
    steps = math.ceil(n / batch_size)
    opt = TypeAdam(model, lr=lr)
    scaler = GraphScaler(model, opt, epochs=epochs, steps_per_epoch=steps, n_train=n)
    scaler.begin()
    model.train()
    for ep in range(epochs):
        if callable(shuffle):
            order = torch.as_tensor(shuffle(ep, n))
        elif shuffle:
            order = torch.randperm(n, generator=generator)
        else:
            order = torch.arange(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size].to(X.device)
            x, t = X[idx], Y[idx]
            y = model(x)
            loss = mse_loss(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scaler.observe(x, y, t)
        scaler.epoch_end()
    scaler.end()
    model.eval()
    return scaler


__all__ += ["fit_graph", "load_snapshot", "LR_SCALE"]
