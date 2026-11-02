"""The graph-type-nn rule, ported line by line from type-nn's ``type_nn_graph.c``.

One :class:`GraphCore` holds the topology (nodes in topological order,
edges with explicit coordinate maps), the slime-mould edge weights, the
feedback accumulators, the evidence cache and every structural decision.
What a node *is* comes from a :class:`NodeStore`:

* :class:`~torch_type_nn.graph.torch_store.TorchStore` wraps an
  :class:`~torch_type_nn.layer.AndOr` (autograd, any optimizer step through
  :class:`~torch_type_nn.optim.TypeAdam`), used by :class:`GraphTypeNN`;
* :class:`~torch_type_nn.cuda.graph.PackedStore` packs the live Ors of a
  node into one ``(R, n_in)`` matrix with a hand-written per-sample backward,
  used by :class:`~torch_type_nn.cuda.graph.CudaGraphTypeNN`.

Everything random is drawn from C's ``xorshift32`` stream in C's order, so a
graph born from a seed has exactly the C graph's weights, and a run follows
the C run as long as floating-point rounding does not flip a decision.

C functions keep their names (``grow_depth``, ``prune_pool``, ...) so the two
files can be read side by side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = ["GraphCore", "Edge", "NodeState", "NodeStore", "X",
           "GROW", "FIT", "PRUNE", "DONE", "phase_at", "birth_depth",
           "xorshift32", "uniform"]

X = -1                       # the model input as an edge source (C: TNNG_X)
GROW, FIT, PRUNE, DONE = 0, 1, 2, 3
MASK = 0xFFFFFFFF
LR_SCALE = 0.1               # common.h TNN_LR_SCALE (the bench.c convention)
_SIZE_T = 1 << 64             # C size_t range: criterion() wraps K like C


def xorshift32(s: int) -> int:
    x = s if s else 2463534242
    x ^= (x << 13) & MASK
    x ^= x >> 17
    x ^= (x << 5) & MASK
    return x & MASK


def uniform(rng: list[int]) -> float:
    """C's tnn_uniform: U(-1, 1) from the xorshift32 state in ``rng[0]``."""
    rng[0] = xorshift32(rng[0])
    return (rng[0] / 4294967296.0) * 2.0 - 1.0


def birth_depth(n_in: int, n_out: int) -> int:
    """round(ln(1 + n m)) with C's lround, at least 1."""
    return max(1, int(math.floor(math.log(1.0 + n_in * n_out) + 0.5)))


def phase_at(u: float) -> int:
    if u < 1.0 / 3.0:
        return GROW
    if u < 2.0 / 3.0:
        return FIT
    return PRUNE


def _rate(acc: float, n: float) -> float:
    return acc / n if n > 0.0 else 0.0


# ---------------------------------------------------------------- the node

class NodeStore:
    """What the rule needs from one node (one type-nn unit layer).

    Ors of unit ``k`` are addressed by their position ``r`` in C's order
    (append on add, order kept on drop). Implementations: ``TorchStore``,
    ``PackedStore``.
    """

    n_in: int
    n_out: int

    def n_or(self, k: int) -> int: ...
    def get_or(self, k: int, r: int) -> tuple[torch.Tensor, float, float]: ...
    def set_or(self, k: int, r: int, w=None, b=None, a=None) -> None: ...
    def or_probe(self, k: int, r: int) -> bool: ...
    def set_or_probe(self, k: int, r: int, v: bool) -> None: ...
    def or_born(self, k: int, r: int) -> int: ...
    def add_or(self, k: int, w: torch.Tensor, b: float, a: float,
               probe: bool, born: int) -> None: ...
    def drop_or(self, k: int, r: int) -> None: ...
    def unit_probe(self, k: int) -> bool: ...
    def set_unit_probe(self, k: int, v: bool) -> None: ...
    def unit_born(self, k: int) -> int: ...
    def set_unit_born(self, k: int, v: int) -> None: ...
    def add_unit(self) -> int: ...
    def drop_unit(self, k: int) -> None: ...
    def add_input(self) -> int: ...
    def drop_input(self, j: int) -> None: ...
    def scale_column(self, j: int, alpha: float, beta: float) -> None: ...
    def cols_get(self, js: list[int]) -> torch.Tensor: ...       # (ors, len(js)), C order
    def cols_set(self, js: list[int], values: torch.Tensor) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def num_params(self) -> int: ...
    def total_ors(self) -> int: ...
    def clone(self) -> "NodeStore": ...


@dataclass
class NodeState:
    store: NodeStore
    probe: bool = False          # the depth dummy
    born: int = 0
    inf_acc: float = 0.0         # lifetime sums: mean |z|, mean |dL/dz|, samples
    bp_acc: float = 0.0
    acc_n: float = 0.0
    ep_inf: float = 0.0          # this epoch
    ep_bp: float = 0.0
    ep_n: float = 0.0

    @property
    def n_in(self) -> int:
        return self.store.n_in

    @property
    def n_out(self) -> int:
        return self.store.n_out


@dataclass
class Edge:
    src: int                     # X = the model input
    dst: int
    pairs: list = field(default_factory=list)   # [k, j]: source coord -> dst column
    in_w: float = 0.0
    out_w: float = 0.0
    c: float = 0.0
    in_acc: float = 0.0
    out_acc: float = 0.0
    grad_acc: float = 0.0
    acc_n: float = 0.0
    ep_in: float = 0.0
    ep_out: float = 0.0
    ep_grad: float = 0.0
    ep_n: float = 0.0
    probe: bool = False          # a merge dummy
    born: int = 0

    def copy(self) -> "Edge":
        e = Edge(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        e.pairs = [list(p) for p in self.pairs]
        return e

    def key(self):
        return (self.dst, self.src if self.src >= 0 else 1 << 62)


@dataclass
class Counters:
    or_add: int = 0
    or_drop: int = 0
    and_add: int = 0
    and_drop: int = 0
    node_add: int = 0
    node_drop: int = 0
    edge_add: int = 0
    edge_drop: int = 0

    # the board's chain-shaped columns
    @property
    def layer_add(self) -> int:
        return self.node_add

    @property
    def layer_drop(self) -> int:
        return self.node_drop

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["layer_add"], d["layer_drop"] = self.node_add, self.node_drop
        return d


# ------------------------------------------------------------- the graph

class GraphCore:
    """Topology, edge weights, feedback, evidence and the dummy rule.

    Subclasses supply ``new_store(n_in, n_out)`` (an empty node with
    ``n_out`` units and no Ors) and, for the hand-written training path,
    ``store.backward_adam``.
    """

    def __init__(self, n_in: int, n_out: int, seed: int, *, device=None,
                 dtype=torch.float64) -> None:
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.rng = [int(seed) & MASK or 1]
        self.nodes: list[NodeState] = []
        self.edges: list[Edge] = []
        self.counters = Counters()
        self.lr = 0.0
        self.step = 0
        self.total = 0
        self.n_train = 0
        self.epochs = 0
        self.phase = GROW
        self.training = False
        self.crit_best = math.inf
        self.loss_sum = 0.0
        self.loss_n = 0
        self.epoch_loss = 0.0
        self.epoch_base = 0.0
        self.t_sum = torch.zeros(self.n_out, dtype=torch.float64, device=self.device)
        self.t_sq = torch.zeros(self.n_out, dtype=torch.float64, device=self.device)
        self._idx_cache: dict = {}
        self.cx: list[torch.Tensor] = []
        self.ct: list[torch.Tensor] = []
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.xin: list[torch.Tensor | None] = []
        self.birth_vertices = 0
        self.birth_edges = 0
        self._col_cache: dict[int, torch.Tensor] = {}
        self._detached = False

    # ----------------------------------------------------------- the store

    def new_store(self, n_in: int, n_out: int) -> NodeStore:  # pragma: no cover
        raise NotImplementedError

    def draw_random(self, n_in: int):
        """C's tnn_or_init_random: w = k U, b = k U, a = 1, k = 1/sqrt(n)."""
        k = 1.0 / math.sqrt(max(n_in, 1))
        w = torch.tensor([k * uniform(self.rng) for _ in range(n_in)], dtype=torch.float64)
        b = k * uniform(self.rng)
        return w, b, 1.0

    def draw_identity(self, n_in: int, noise: float):
        """C's tnn_or_init_identity: w = noise U, b = 1 + noise U, a = 1."""
        w = torch.tensor([noise * uniform(self.rng) for _ in range(n_in)], dtype=torch.float64)
        b = 1.0 + noise * uniform(self.rng)
        return w, b, 1.0

    # -------------------------------------------------------------- birth

    def build(self) -> None:
        """tnng_create: the backbone with the chain's init law, then the
        small-world shortcuts born as dummies."""
        D = birth_depth(self.n_in, self.n_out)
        self.birth_vertices = D
        n = self.n_in
        for i in range(D):
            node = self._node_insert(i, n, self.n_out)
            for k in range(self.n_out):
                w, b, a = self.draw_random(n)
                node.store.add_or(k, w, b, a, False, 0)
            e = Edge(i - 1 if i else X, i, [[j, j] for j in range(n)],
                     in_w=1.0, out_w=1.0, c=1.0, born=self.step)
            self.edges.append(e)
            n = self.n_out
        self.sort_edges()
        H = int(math.ceil(math.log(D))) if D > 1 else 0
        for i in range(D):
            for h in range(2, H + 1):
                if i + h >= D:
                    continue
                t = i + h
                self.rng[0] = xorshift32(self.rng[0])
                r = self.rng[0] / 4294967296.0
                if r < 1.0 / D and D > i + 2:
                    span = D - (i + 2)
                    self.rng[0] = xorshift32(self.rng[0])
                    t = i + 2 + (self.rng[0] % span)
                self.add_dummy_edge(i, t)
        self.birth_edges = len(self.edges)

    # ---------------------------------------------------------- bookkeeping

    def _node_insert(self, pos: int, n_in: int, n_out: int) -> NodeState:
        node = NodeState(self.new_store(n_in, n_out))
        self.nodes.insert(pos, node)
        self.xin.insert(pos, None)
        for e in self.edges:
            if e.src != X and e.src >= pos:
                e.src += 1
            if e.dst >= pos:
                e.dst += 1
        self.on_structure()
        return node

    def _node_remove(self, pos: int) -> None:
        del self.nodes[pos]
        del self.xin[pos]
        for e in self.edges:
            if e.src != X and e.src > pos:
                e.src -= 1
            if e.dst > pos:
                e.dst -= 1
        self.on_structure()

    def on_structure(self) -> None:
        """Hook for the module wrapper (re-registers node parameters)."""
        self._col_cache = {}

    def sort_edges(self) -> None:
        self.edges.sort(key=Edge.key)

    def edge_find(self, src: int, dst: int) -> int:
        for i, e in enumerate(self.edges):
            if e.src == src and e.dst == dst:
                return i
        return -1

    def in_degree(self, v: int) -> int:
        return sum(1 for e in self.edges if e.dst == v)

    def out_degree(self, u: int) -> int:
        return sum(1 for e in self.edges if e.src == u)

    def reads_x(self, v: int) -> bool:
        return any(e.dst == v and e.src == X for e in self.edges)

    def src_width(self, src: int) -> int:
        return self.n_in if src == X else self.nodes[src].n_out

    def depth_probe(self) -> int:
        for i, n in enumerate(self.nodes):
            if n.probe:
                return i
        return -1

    def width_probe_unit(self, u: int) -> int:
        st = self.nodes[u].store
        for k in range(st.n_out):
            if st.unit_probe(k):
                return k
        return -1

    def dst_ors(self, v: int) -> int:
        return self.nodes[v].store.total_ors()

    # ---------------------------------------------------- columns and coords

    def add_column(self, v: int) -> int:
        return self.nodes[v].store.add_input()

    def drop_column(self, v: int, j: int) -> None:
        self.nodes[v].store.drop_input(j)
        for e in self.edges:
            if e.dst != v:
                continue
            e.pairs = [[k, jj - 1 if jj > j else jj] for k, jj in e.pairs if jj != j]

    def drop_coord(self, u: int, k: int) -> None:
        for e in self.edges:
            if e.src != u:
                continue
            while True:
                hit = next((p for p in e.pairs if p[0] == k), None)
                if hit is None:
                    break
                self.drop_column(e.dst, hit[1])
            for p in e.pairs:
                if p[0] > k:
                    p[0] -= 1
        self.nodes[u].store.drop_unit(k)

    def remove_edge(self, i: int) -> None:
        e = self.edges[i]
        while e.pairs:
            self.drop_column(e.dst, e.pairs[-1][1])
        del self.edges[i]

    # ----------------------------------------------- edge weights (slime mould)

    def renormalize(self) -> None:
        """Shares from the lifetime-averaged feedback; the destination absorbs
        every change of c (w' = w c / c') so the function does not move."""
        E = self.edges
        rin = [_rate(e.in_acc, e.acc_n) for e in E]
        rout = [_rate(e.out_acc, e.acc_n) for e in E]
        new_in, new_out = [], []
        for a in E:
            si = so = 0.0
            li = lo = di = do = 0
            for i, b in enumerate(E):
                if b.dst == a.dst:
                    si += rin[i]
                    di += 1
                    li += not b.probe
                if b.src == a.src:
                    so += rout[i]
                    do += 1
                    lo += not b.probe
            new_in.append(_rate(_rate(a.in_acc, a.acc_n), si) if si > 0.0
                          else ((0.0 if a.probe else 1.0 / li) if li else 1.0 / di))
            new_out.append(_rate(_rate(a.out_acc, a.acc_n), so) if so > 0.0
                           else ((0.0 if a.probe else 1.0 / lo) if lo else 1.0 / do))
        for a, ni, no in zip(E, new_in, new_out, strict=True):
            cn, co = ni * no, a.c
            a.in_w, a.out_w = ni, no
            if co > 0.0 and cn > 0.0 and cn != co:
                st = self.nodes[a.dst].store
                for _, j in a.pairs:
                    st.scale_column(j, co / cn, 0.0)
                a.c = cn
            elif co > 0.0 and cn == 0.0:
                pass             # keeps its gain until pruned (C: same)
            else:
                a.c = cn
        self.on_gains()

    def on_gains(self) -> None:
        """Hook: gains changed (the module refreshes per-column Adam scales)."""
        self._col_cache = {}

    def col_lr(self, v: int) -> torch.Tensor:
        """Adam scale per input column of node v: 1/c of the wrapping edge."""
        s = torch.ones(self.nodes[v].n_in, dtype=torch.float64)
        for e in self.edges:
            if e.dst == v:
                sc = 1.0 / e.c if e.c > 0.0 else 1.0
                for _, j in e.pairs:
                    s[j] = sc
        return s

    def fold_feedback(self) -> None:
        for e in self.edges:
            e.in_acc += float(e.ep_in)
            e.out_acc += float(e.ep_out)
            e.grad_acc += float(e.ep_grad)
            e.acc_n += float(e.ep_n)
            e.ep_in = e.ep_out = e.ep_grad = e.ep_n = 0.0
        for n in self.nodes:
            n.inf_acc += float(n.ep_inf)
            n.bp_acc += float(n.ep_bp)
            n.acc_n += float(n.ep_n)
            n.ep_inf = n.ep_bp = n.ep_n = 0.0

    # ------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor, *, hooks: bool = False) -> torch.Tensor:
        """Batched forward. ``x`` is ``(B, n_in)``. With ``self.training`` the
        inference feedback is accumulated per sample (C: per forward); with
        ``hooks`` the back-prop feedback is accumulated by autograd hooks."""
        B = x.shape[0]
        zs: list[torch.Tensor | None] = [None] * len(self.nodes)
        by_dst: dict[int, list[Edge]] = {}
        for e in self.edges:
            by_dst.setdefault(e.dst, []).append(e)
        train = self.training
        for v, node in enumerate(self.nodes):
            ins = by_dst.get(v, [])
            parts, cols = [], []
            for e in ins:
                src = x if e.src == X else zs[e.src]
                if not e.pairs:
                    continue
                ks, _ = self._idx(e, x.device)
                val = src.index_select(1, ks)
                if train:
                    e.ep_in = e.ep_in + val.detach().abs().mean(1).sum()
                parts.append(e.c * val)
                cols.extend(p[1] for p in e.pairs)
            if train:
                for e in ins:
                    e.ep_n += B
            n_in = node.n_in
            if parts:
                cat = torch.cat(parts, 1)
                perm = torch.empty(n_in, dtype=torch.long, device=x.device)
                perm[torch.tensor(cols, device=x.device)] = torch.arange(len(cols),
                                                                         device=x.device)
                xin = cat.index_select(1, perm) if len(cols) == n_in else \
                    _scatter_cols(cat, cols, n_in)
            else:
                xin = x.new_zeros(B, n_in)
            if hooks and xin.requires_grad:
                xin.register_hook(self._grad_hook(ins))
            self.xin[v] = xin
            z = node.store.forward(xin)
            if train:
                if node.n_out:
                    node.ep_inf = node.ep_inf + z.detach().abs().mean(1).sum()
                node.ep_n += B
            if hooks and z.requires_grad:
                z.register_hook(self._bp_hook(node, ins))
            zs[v] = z
        return zs[-1]

    def _idx(self, e: Edge, device):
        """(source coords, destination columns) of an edge as index tensors,
        cached by content so a training step makes no host-to-device copy."""
        key = (str(device), tuple(map(tuple, e.pairs)))
        t = self._idx_cache.get(key)
        if t is None:
            if len(self._idx_cache) > 4096:
                self._idx_cache.clear()
            t = (torch.tensor([p[0] for p in e.pairs], dtype=torch.long, device=device),
                 torch.tensor([p[1] for p in e.pairs], dtype=torch.long, device=device))
            self._idx_cache[key] = t
        return t

    def _grad_hook(self, ins):
        def hook(dx: torch.Tensor) -> None:
            with torch.no_grad():
                for e in ins:
                    if e.pairs:
                        _, js = self._idx(e, dx.device)
                        e.ep_grad = e.ep_grad + dx.index_select(1, js).abs().mean(1).sum()
        return hook

    @staticmethod
    def _bp_hook(node: NodeState, ins):
        def hook(gz: torch.Tensor) -> None:
            with torch.no_grad():
                bp = gz.abs().mean(1).sum() if gz.shape[1] else 0.0
                node.ep_bp = node.ep_bp + bp
                for e in ins:
                    e.ep_out = e.ep_out + bp
        return hook

    # ------------------------------------------- observing a training step

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> None:
        """Residual statistics and the evidence cache of one optimizer step
        (C: the training half of tnng_backward), then step += 1."""
        with torch.no_grad():
            y64 = y.detach().to(self.t_sum).reshape(-1, self.n_out)
            t64 = t.detach().to(self.t_sum).reshape(-1, self.n_out)
            r = y64 - t64
            self.loss_sum = self.loss_sum + (r * r).mean(1).sum()
            self.loss_n += y64.shape[0]
            self.t_sum += t64.sum(0)
            self.t_sq += (t64 * t64).sum(0)
            self.cx.append(x.detach().reshape(-1, self.n_in))
            self.ct.append(t.detach().reshape(-1, self.n_out))
            self._cache = None
        self.step += 1

    def _param_device(self):
        """Device of the live parameters (may differ from the birth device)."""
        for n in self.nodes:
            store = getattr(n, "store", None)
            if store is not None and hasattr(store, "L"):
                return store.L.weight.device
        return self.device

    def cache(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.cx:
            return None
        if self._cache is None:
            dev = self._param_device()
            self._cache = (torch.cat(self.cx).to(dev, self.dtype),
                           torch.cat(self.ct).to(dev, self.dtype))
        return self._cache

    @property
    def cache_n(self) -> int:
        return sum(c.shape[0] for c in self.cx)

    def clear_cache(self) -> None:
        self.cx, self.ct, self._cache = [], [], None

    # ----------------------------------------------------------- evidence

    @torch.no_grad()
    def measure(self) -> float:
        c = self.cache()
        if c is None:
            return self.epoch_loss
        tr, self.training = self.training, False
        y = self.forward(c[0])
        self.training = tr
        return float(((y - c[1]) ** 2).mean())

    def n_obs(self) -> float:
        n = self.cache_n or self.n_train
        return float(n) * float(self.n_out)

    def bic_ratio(self, with_: float, without: float, k: int) -> float:
        n = self.n_obs()
        if k == 0 or n <= 1.0:
            return math.inf
        if not with_ > 0.0:
            return math.inf if without > 0.0 else 0.0
        if without <= with_:
            return 0.0
        return n * math.log(without / with_) / (k * math.log(n))

    def criterion(self, mse: float, K: int) -> float:
        # C passes K as size_t. In prune_pool, ``P - K - np`` can go below 0:
        # a coordinate's np also counts the parameters of Ors already counted
        # in K. C then wraps to ~1.8e19, which rejects every later candidate.
        # A Python int would go negative and accept them all, pruning far
        # past C. Wrap the same way, so the port takes C's decisions.
        K %= _SIZE_T
        n = self.n_obs()
        lm = math.log(mse) if mse > 0.0 else -math.inf
        return n * lm + float(K) * math.log(n if n > 1.0 else 2.0)

    def num_params(self) -> int:
        return sum(n.store.num_params() for n in self.nodes)

    def _cols_ablate(self, v: int, js: list[int]):
        st = self.nodes[v].store
        snap = st.cols_get(js)
        st.cols_set(js, torch.zeros_like(snap))
        return (v, list(js), snap)

    def _cols_restore(self, s) -> None:
        v, js, snap = s
        self.nodes[v].store.cols_set(js, snap)

    def _cols_rms(self, v: int, js: list[int]) -> float:
        w = self.nodes[v].store.cols_get(js)
        return math.sqrt(float((w * w).mean())) if w.numel() else 0.0

    def _coord_ablate(self, u: int, k: int):
        snaps, P, ss, cc = [], 0, 0.0, 0
        for e in self.edges:
            if e.src != u:
                continue
            js = [j for kk, j in e.pairs if kk == k]
            if js:
                r = self._cols_rms(e.dst, js)
                no = self.dst_ors(e.dst) * len(js)
                ss += r * r * no
                cc += no
                P += no
                snaps.append(self._cols_ablate(e.dst, js))
        return snaps, P, (math.sqrt(ss / cc) if cc else 0.0)

    def _or_ablate(self, v: int, k: int, r: int):
        st = self.nodes[v].store
        w, b, a = st.get_or(k, r)
        st.set_or(k, r, torch.zeros_like(w), 1.0, 1.0)
        return (v, k, r, w, b, a)

    def _or_restore(self, s) -> None:
        v, k, r, w, b, a = s
        self.nodes[v].store.set_or(k, r, w, b, a)

    def evidence_or(self, base: float, v: int, k: int, r: int) -> float:
        s = self._or_ablate(v, k, r)
        without = self.measure()
        self._or_restore(s)
        return self.bic_ratio(base, without, self.nodes[v].n_in + 2)

    def unit_params(self, u: int, k: int) -> int:
        st = self.nodes[u].store
        return st.n_or(k) * (st.n_in + 2)

    def evidence_coord(self, base: float, u: int, k: int) -> float:
        snaps, P, _ = self._coord_ablate(u, k)
        without = self.measure()
        for s in snaps:
            self._cols_restore(s)
        return self.bic_ratio(base, without, P + self.unit_params(u, k))

    def evidence_edge(self, base: float, i: int) -> float:
        e = self.edges[i]
        js = [j for _, j in e.pairs]
        s = self._cols_ablate(e.dst, js)
        without = self.measure()
        self._cols_restore(s)
        return self.bic_ratio(base, without, self.dst_ors(e.dst) * len(js))

    @staticmethod
    def _sq_one(w, b, a) -> float:
        return (b - 1.0) ** 2 + (a - 1.0) ** 2 + float((w * w).sum())

    @staticmethod
    def _sq_carrier(w, b, a, k) -> float:
        d = w.clone()
        d[k] -= 1.0
        return b * b + (a - 1.0) ** 2 + float((d * d).sum())

    def dev_or(self, v: int, k: int, r: int) -> float:
        st = self.nodes[v].store
        w, b, a = st.get_or(k, r)
        return math.sqrt(self._sq_one(w, b, a) / (st.n_in + 2))

    def dev_layer(self, v: int) -> float:
        """type_nn_scale.c tnn_dev_layer."""
        st = self.nodes[v].store
        if st.n_in != st.n_out:
            return math.inf
        n, s, c = st.n_in, 0.0, 0
        for k in range(st.n_out):
            ors = [st.get_or(k, r) for r in range(st.n_or(k))]
            all_one = sum(self._sq_one(*o) for o in ors)
            best = math.inf
            for o in ors:
                val = all_one - self._sq_one(*o) + self._sq_carrier(*o, k)
                best = min(best, val)
            s += best
            c += len(ors) * (n + 2)
        return math.sqrt(s / c) if c else math.inf

    def evidence_node(self, base: float, v: int) -> float:
        st = self.nodes[v].store
        if st.n_in != st.n_out:
            return math.inf
        n, saved, cnt = st.n_in, [], 0
        for k in range(st.n_out):
            ors = [st.get_or(k, r) for r in range(st.n_or(k))]
            cnt += len(ors)
            best, bd = 0, math.inf
            for r, o in enumerate(ors):
                d = self._sq_carrier(*o, k) - self._sq_one(*o)
                if d < bd:
                    bd, best = d, r
            for r, (w, b, a) in enumerate(ors):
                saved.append((k, r, w, b, a))
                nw = torch.zeros_like(w)
                if r == best:
                    nw[k] = 1.0
                    st.set_or(k, r, nw, 0.0, 1.0)
                else:
                    st.set_or(k, r, nw, 1.0, 1.0)
        without = self.measure()
        for k, r, w, b, a in saved:
            st.set_or(k, r, w, b, a)
        return self.bic_ratio(base, without, cnt * (n + 2))

    # --------------------------------------------------------------- folds

    def clone(self) -> "GraphCore":
        """Structure and weights; the clone borrows the training cache."""
        h = object.__new__(type(self))
        h.__dict__.update(self.__dict__)
        h.nodes = [NodeState(**{**n.__dict__, "store": n.store.clone()}) for n in self.nodes]
        h.edges = [e.copy() for e in self.edges]
        h.xin = [None] * len(h.nodes)
        h.counters = Counters(**self.counters.__dict__)
        h.rng = self.rng            # one stream (C: the clone's rng is never drawn)
        h.training = False
        h._detached = True          # a clone never touches the module wrapper
        h._col_cache = {}
        return h

    def adopt(self, h: "GraphCore") -> None:
        self.nodes, self.edges = h.nodes, h.edges
        self.xin = [None] * len(self.nodes)
        self.on_structure()
        self.on_gains()

    @torch.no_grad()
    def fold_cols(self, old: "GraphCore", fc: list[tuple[int, int, int]]) -> None:
        c = self.cache()
        if not fc or c is None:
            return
        tr, to = self.training, old.training
        self.training = old.training = False
        old.forward(c[0])
        self.forward(c[0])
        self.training, old.training = tr, to
        N = c[0].shape[0]
        fits = []
        for v_new, v_old, j in fc:
            a = old.xin[v_old][:, j].to(torch.float64)
            b = self.xin[v_new][:, j].to(torch.float64)
            ma, mb = float(a.sum()) / N, float(b.sum()) / N
            var = float((b * b).sum()) / N - mb * mb
            cov = float((a * b).sum()) / N - ma * mb
            al = cov / var if var > 0.0 else 1.0
            if not al > 0.0 or not math.isfinite(al):
                al = 1.0
            be = ma - al * mb
            if not math.isfinite(be):
                be = 0.0
            fits.append((v_new, j, al, be))
        for v_new, j, al, be in fits:
            self.nodes[v_new].store.scale_column(j, al, be)

    # ------------------------------------------------------------- dummies

    def add_dummy_edge(self, u: int, v: int) -> bool:
        if u == X or u >= v or v >= len(self.nodes):
            return False
        if self.reads_x(v) or self.edge_find(u, v) >= 0:
            return False
        if self.nodes[u].probe or self.nodes[v].probe:
            return False
        e = Edge(u, v, [], c=0.0, probe=True, born=self.step)
        for k in range(self.nodes[u].n_out):
            e.pairs.append([k, self.add_column(v)])
        self.edges.append(e)
        self.sort_edges()
        return True

    def add_width_probe(self, u: int) -> bool:
        if u + 1 >= len(self.nodes) or self.nodes[u].probe:
            return False
        if not any(e.src == u and not self.nodes[e.dst].probe for e in self.edges):
            return False
        st = self.nodes[u].store
        k = st.add_unit()
        w, b, a = self.draw_random(st.n_in)
        st.add_or(k, w, b, a, False, self.step)
        st.set_unit_probe(k, True)
        st.set_unit_born(k, self.step)
        for e in self.edges:
            if e.src != u or self.nodes[e.dst].probe:
                continue
            e.pairs.append([k, self.add_column(e.dst)])
        return True

    def insert_node(self, ei: int) -> bool:
        if ei >= len(self.edges) or self.depth_probe() >= 0:
            return False
        E = self.edges[ei]
        u, v = E.src, E.dst
        if E.probe or self.nodes[v].probe or (u != X and self.nodes[u].probe):
            return False
        wp = self.width_probe_unit(u) if u != X else -1
        cp = [list(p) for p in E.pairs if p[0] != wp]
        p = len(cp)
        if not p:
            return False
        old = self.clone()
        E.pairs = [list(q) for q in E.pairs if q[0] == wp]
        acc = (E.in_acc, E.out_acc, E.grad_acc, E.acc_n)
        pos = 0 if u == X else u + 1
        w = self._node_insert(pos, p, p)
        vn = v + 1
        for i in range(p):
            wv = torch.tensor([self.lr * uniform(self.rng) for _ in range(p)],
                              dtype=torch.float64)
            wv[i] += 1.0
            bb = self.lr * uniform(self.rng)
            w.store.add_or(i, wv, bb, 1.0, False, self.step)
        w.probe, w.born = True, self.step
        w.inf_acc, w.bp_acc, w.acc_n = acc[0], acc[1], acc[3]
        a = Edge(u, pos, [[q[0], i] for i, q in enumerate(cp)], born=self.step)
        f = Edge(pos, vn, [[i, q[1]] for i, q in enumerate(cp)], born=self.step)
        for ed in (a, f):
            ed.in_acc, ed.out_acc, ed.grad_acc, ed.acc_n = acc
        self.edges.append(a)
        self.edges.append(f)
        if not E.pairs:
            self.edges = [e for e in self.edges if e is not E]
        self.sort_edges()
        self.renormalize()
        self.fold_cols(old, [(vn, v, q[1]) for q in cp])
        return True

    def remove_node(self, w: int) -> bool:
        if w + 1 >= len(self.nodes):
            return False
        st = self.nodes[w].store
        if st.n_in != st.n_out:
            return False
        ins = [i for i, e in enumerate(self.edges) if e.dst == w]
        if len(ins) != 1:
            return False
        a = self.edges[ins[0]]
        u, n = a.src, st.n_in
        mp = [None] * n
        for k, j in a.pairs:
            mp[j] = k
        if any(m is None for m in mp):
            return False
        if u == X:
            for e in self.edges:
                if e.src == w and self.in_degree(e.dst) != 1:
                    return False
        old = self.clone()
        fc = []
        outs = [e for e in self.edges if e.src == w]
        for F in outs:
            v = F.dst
            hi = self.edge_find(u, v)
            if hi < 0:
                Hh = Edge(u, v, [], c=0.0, born=F.born)
                Hh.in_acc, Hh.out_acc, Hh.grad_acc, Hh.acc_n = (F.in_acc, F.out_acc,
                                                                F.grad_acc, F.acc_n)
                self.edges.append(Hh)
            else:
                Hh = self.edges[hi]
            Hh.probe = False
            for k, j in F.pairs:
                Hh.pairs.append([mp[k], j])
                fc.append((v - 1, v, j))
            F.pairs = []
        self.edges = [e for e in self.edges if e.dst != w and e.src != w]
        self._node_remove(w)
        self.sort_edges()
        self.renormalize()
        self.fold_cols(old, fc)
        return True

    # ---------------------------------------------------------------- grow

    def theta_up(self, age: int) -> float:
        return self.lr * float(max(age, 1)) ** 0.75

    def grow_and(self, base: float) -> None:
        for v, node in enumerate(self.nodes):
            st = node.store
            for k in range(st.n_out):
                r = next((q for q in range(st.n_or(k)) if st.or_probe(k, q)), -1)
                if r >= 0:
                    if (self.dev_or(v, k, r) > self.theta_up(self.step - st.or_born(k, r))
                            and self.evidence_or(base, v, k, r) > 1.0):
                        st.set_or_probe(k, r, False)
                        self.counters.and_add += 1
                        r = -1
                if r < 0:
                    w, b, a = self.draw_identity(st.n_in, self.lr)
                    st.add_or(k, w, b, a, True, self.step)

    def grow_width(self, base: float) -> None:
        for u in range(len(self.nodes) - 1):
            if self.nodes[u].probe:
                continue
            k = self.width_probe_unit(u)
            if k >= 0:
                snaps, _, rms = self._coord_ablate(u, k)
                for s in snaps:
                    self._cols_restore(s)
                age = self.step - self.nodes[u].store.unit_born(k)
                if rms > self.theta_up(age) and self.evidence_coord(base, u, k) > 1.0:
                    self.nodes[u].store.set_unit_probe(k, False)
                    self.counters.or_add += 1
                    k = -1
            if k < 0:
                self.add_width_probe(u)

    def best_source(self, v: int) -> int:
        best, br = -1, -1.0
        for u in range(v):
            if self.nodes[u].probe or self.edge_find(u, v) >= 0:
                continue
            r = _rate(self.nodes[u].inf_acc, self.nodes[u].acc_n)
            if r > br:
                br, best = r, u
        return best

    def grow_edges(self, base: float) -> None:
        v = 1
        while v < len(self.nodes):
            if self.nodes[v].probe or self.reads_x(v):
                v += 1
                continue
            p = next((i for i, e in enumerate(self.edges) if e.dst == v and e.probe), -1)
            s = self.best_source(v)
            if p >= 0:
                E = self.edges[p]
                rms = self._cols_rms(v, [j for _, j in E.pairs])
                if (rms > self.theta_up(self.step - E.born)
                        and self.evidence_edge(base, p) > 1.0):
                    E.probe = False
                    self.counters.edge_add += 1
                    s = self.best_source(v)
                elif s >= 0 and (_rate(self.nodes[s].inf_acc, self.nodes[s].acc_n)
                                 > _rate(self.nodes[E.src].inf_acc, self.nodes[E.src].acc_n)):
                    self.remove_edge(p)
                    self.add_dummy_edge(s, v)
                    v += 1
                    continue
                else:
                    v += 1
                    continue
            if s >= 0:
                self.add_dummy_edge(s, v)
            v += 1

    def loudest_edge(self) -> tuple[int, float]:
        best, bs = -1, -1.0
        for i, e in enumerate(self.edges):
            if e.probe or self.nodes[e.dst].probe:
                continue
            if e.src != X and self.nodes[e.src].probe:
                continue
            r = _rate(e.grad_acc, e.acc_n)
            if r > bs:
                bs, best = r, i
        return best, bs

    def probe_score(self, w: int) -> float:
        s = -1.0
        for e in self.edges:
            if e.dst == w or e.src == w:
                s = max(s, _rate(e.grad_acc, e.acc_n))
        return s

    def grow_depth(self) -> None:
        w = self.depth_probe()
        if w >= 0:
            n = self.nodes[w]
            age = self.step - n.born
            if (self.dev_layer(w) > self.theta_up(age)
                    and self.evidence_node(self.measure(), w) > 1.0):
                n.probe = False
                self.counters.node_add += 1
            else:
                e, ls = self.loudest_edge()
                if e < 0 or ls <= self.probe_score(w):
                    return
                if not self.remove_node(w):
                    return
        e, _ = self.loudest_edge()
        if e >= 0:
            self.insert_node(e)

    # --------------------------------------------------------------- prune

    def cleanup(self) -> None:
        changed = True
        while changed:
            changed = False
            if any(not e.pairs for e in self.edges):
                self.edges = [e for e in self.edges if e.pairs]
                changed = True
            u = 0
            while u + 1 < len(self.nodes):
                if self.out_degree(u) == 0:
                    self.edges = [e for e in self.edges if e.dst != u]
                    if not self.nodes[u].probe:
                        self.counters.node_drop += 1
                    self._node_remove(u)
                    changed = True
                    break
                st = self.nodes[u].store
                k = st.n_out - 1
                while k >= 0 and st.n_out > 1:
                    read = any(e.src == u and any(p[0] == k for p in e.pairs)
                               for e in self.edges)
                    if not read:
                        if not st.unit_probe(k):
                            self.counters.or_drop += 1
                        self.drop_coord(u, k)
                        changed = True
                    k -= 1
                u += 1
        self.sort_edges()

    def try_remove(self, w: int, ref: float) -> bool:
        h = self.clone()
        if not h.remove_node(w):
            return False
        if self.criterion(h.measure(), h.num_params()) <= ref:
            self.adopt(h)
            return True
        return False

    def prune_depth(self, ref: float) -> None:
        p = self.depth_probe()
        if p >= 0:
            if self.try_remove(p, ref):
                return
            self.nodes[p].probe = False
            self.counters.node_add += 1
        for w in range(len(self.nodes) - 1):
            if self.try_remove(w, ref):
                self.counters.node_drop += 1
                return

    def prune_edges(self, ref: float) -> None:
        keys = sorted(((e.c, e.dst, e.src) for e in self.edges if e.src != X))
        P = self.num_params()
        for _c, dst, src in keys:
            i = self.edge_find(src, dst)
            if i < 0:
                continue
            E = self.edges[i]
            probe = E.probe
            if self.in_degree(E.dst) < 2:
                if probe:
                    E.probe = False
                    self.counters.edge_add += 1
                continue
            js = [j for _, j in E.pairs]
            bp = self.dst_ors(E.dst) * len(js)
            s = self._cols_ablate(E.dst, js)
            m = self.measure()
            if self.criterion(m, P - bp) <= ref:
                self.remove_edge(i)
                P -= bp
                if not probe:
                    self.counters.edge_drop += 1
            else:
                self._cols_restore(s)
                if probe:
                    E.probe = False
                    self.counters.edge_add += 1

    def column_edge(self, v: int, j: int) -> tuple[int, int]:
        for i, e in enumerate(self.edges):
            if e.dst != v:
                continue
            for k, jj in e.pairs:
                if jj == j:
                    return i, k
        return -1, -1

    def prune_pool(self, ref: float) -> None:
        base = self.measure()
        P = self.num_params()
        cs = []                     # (alone, seq, kind, v, k, r, np)
        for v, node in enumerate(self.nodes):
            st = node.store
            for k in range(st.n_out):
                for r in range(st.n_or(k)):
                    s = self._or_ablate(v, k, r)
                    without = self.measure()
                    self._or_restore(s)
                    cs.append([without - base, len(cs), 0, v, k, r, st.n_in + 2])
            if v + 1 < len(self.nodes):
                for k in range(st.n_out):
                    snaps, pp, _ = self._coord_ablate(v, k)
                    without = self.measure()
                    for s in snaps:
                        self._cols_restore(s)
                    cs.append([without - base, len(cs), 1, v, k, 0,
                               pp + self.unit_params(v, k)])
            for j in range(st.n_in):
                e, _ = self.column_edge(v, j)
                if e < 0 or self.edges[e].src == X or len(self.edges[e].pairs) < 2:
                    continue
                s = self._cols_ablate(v, [j])
                without = self.measure()
                self._cols_restore(s)
                cs.append([without - base, len(cs), 2, v, j, 0, self.dst_ors(v)])
        cs.sort(key=lambda c: (c[0], c[1]))

        NN = len(self.nodes)
        or_gone = {(v, k): set() for v in range(NN) for k in range(self.nodes[v].n_out)}
        or_left = {(v, k): self.nodes[v].store.n_or(k)
                   for v in range(NN) for k in range(self.nodes[v].n_out)}
        unit_gone: set = set()
        coord_left = [self.nodes[v].n_out for v in range(NN)]
        col_gone: set = set()
        edge_left = [len(e.pairs) for e in self.edges]
        K = 0
        for c in cs:
            _, _, kind, v, k, r, np_ = c
            if kind == 0:
                if (v, k) in unit_gone or or_left[(v, k)] <= 1:
                    continue
                s = self._or_ablate(v, k, r)
                if self.criterion(self.measure(), P - K - np_) > ref:
                    self._or_restore(s)
                    continue
                or_gone[(v, k)].add(r)
                or_left[(v, k)] -= 1
            elif kind == 1:
                if (v, k) in unit_gone or coord_left[v] <= 1:
                    continue
                unsafe = False
                for i, E in enumerate(self.edges):
                    if E.src != v:
                        continue
                    mine = sum(1 for kk, j in E.pairs if kk == k and (E.dst, j) not in col_gone)
                    if mine and mine >= edge_left[i] and self.in_degree(E.dst) < 2:
                        unsafe = True
                        break
                if unsafe:
                    continue
                snaps, pp, _ = self._coord_ablate(v, k)
                np_ = pp + self.unit_params(v, k)
                if self.criterion(self.measure(), P - K - np_) > ref:
                    for s in snaps:
                        self._cols_restore(s)
                    continue
                unit_gone.add((v, k))
                coord_left[v] -= 1
                for i, E in enumerate(self.edges):
                    if E.src != v:
                        continue
                    for kk, j in E.pairs:
                        if kk == k and (E.dst, j) not in col_gone:
                            col_gone.add((E.dst, j))
                            edge_left[i] -= 1
            else:
                j = k
                if (v, j) in col_gone:
                    continue
                e, kk = self.column_edge(v, j)
                if e < 0 or edge_left[e] <= 1:
                    continue
                if (self.edges[e].src, kk) in unit_gone:
                    continue
                s = self._cols_ablate(v, [j])
                if self.criterion(self.measure(), P - K - np_) > ref:
                    self._cols_restore(s)
                    continue
                col_gone.add((v, j))
                edge_left[e] -= 1
            K += np_

        # apply: Ors, then columns (descending), then coordinates (descending)
        for v in range(NN):
            st = self.nodes[v].store
            for k in range(st.n_out):
                for r in range(st.n_or(k) - 1, -1, -1):
                    was = st.or_probe(k, r)
                    if r in or_gone[(v, k)] and (v, k) not in unit_gone:
                        st.drop_or(k, r)
                        if not was:
                            self.counters.and_drop += 1
                    elif was and (v, k) not in unit_gone:
                        st.set_or_probe(k, r, False)
                        self.counters.and_add += 1
        for v in range(NN):
            for j in range(self.nodes[v].n_in - 1, -1, -1):
                if (v, j) in col_gone:
                    self.drop_column(v, j)
        for v in range(NN - 1, -1, -1):
            st = self.nodes[v].store
            for k in range(st.n_out - 1, -1, -1):
                was = st.unit_probe(k)
                if (v, k) in unit_gone:
                    self.drop_coord(v, k)
                    if not was:
                        self.counters.or_drop += 1
                elif was:
                    st.set_unit_probe(k, False)
                    self.counters.or_add += 1

    def prune_step(self) -> None:
        cur = self.criterion(self.measure(), self.num_params())
        self.crit_best = min(self.crit_best, cur)
        self.prune_depth(self.crit_best)
        self.prune_edges(self.crit_best)
        self.prune_pool(self.crit_best)
        self.cleanup()
        self.renormalize()
        cur = self.criterion(self.measure(), self.num_params())
        self.crit_best = min(self.crit_best, cur)

    # ------------------------------------------------------------ schedule

    def begin(self, n_train: int, epochs: int, lr: float, *, total: int | None = None) -> None:
        """``lr`` is the effective step (bench.c: task lr x 0.1); ``total`` the
        optimizer steps of the run (default n_train x epochs, per sample)."""
        self.n_train, self.epochs = int(n_train), int(epochs)
        self.lr = float(lr)
        self.step = 0
        self.total = int(total if total is not None else n_train * epochs)
        self.phase = GROW
        self.crit_best = math.inf
        self.t_sum.zero_()
        self.t_sq.zero_()
        self.grow_width(0.0)
        self.grow_and(0.0)
        self.clear_cache()
        self.on_gains()

    def close_residual(self) -> None:
        n, m = self.loss_n, self.n_out
        self.epoch_loss = float(self.loss_sum) / n if n else 0.0
        base = 0.0
        if n:
            mu = self.t_sum / n
            var = self.t_sq / n - mu * mu
            base = float(var.clamp(min=0.0).sum())
        self.epoch_base = base / m if m else 0.0
        self.t_sum.zero_()
        self.t_sq.zero_()
        self.loss_sum = 0.0
        self.loss_n = 0

    def epoch_end(self) -> None:
        if self.phase == DONE:
            return
        u = self.step / self.total if self.total else 1.0
        ph = phase_at(u)
        self.close_residual()
        self.fold_feedback()
        self.renormalize()
        n = float(self.n_train or 1)
        unexplained = self.epoch_loss > self.epoch_base / n
        if ph == GROW and unexplained:
            self.grow_depth()
            base = self.measure()
            self.grow_width(base)
            self.grow_edges(base)
            self.grow_and(base)
        elif ph == PRUNE:
            self.prune_step()
        self.phase = ph
        self.clear_cache()
        self.on_gains()

    def end(self) -> None:
        if self.phase not in (PRUNE, DONE):
            self.prune_step()
        self.phase = DONE
        self.on_gains()

    # ----------------------------------------- the hand-written training path

    def col_lr_cached(self, v: int) -> torch.Tensor:
        t = self._col_cache.get(v)
        if t is None:
            t = self.col_lr(v).to(self.device, self.dtype)
            self._col_cache[v] = t
        return t

    def step_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One bench.c sample with the per-Or Adam of the store
        (``store.backward_adam``): forward, mean-MSE dy = (y - t)/m, backward."""
        x = x.reshape(1, self.n_in).to(self.device, self.dtype)
        t = t.reshape(-1).to(self.device, self.dtype)
        y = self.forward(x)[0]
        dy = (y - t) / self.n_out
        self.backward_manual(x, y, dy)
        return y

    @torch.no_grad()
    def backward_manual(self, x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor) -> None:
        """tnng_backward for one sample."""
        m = self.n_out
        if self.training:
            r = m * dy
            tt = y - r
            self.loss_sum = self.loss_sum + (r * r).sum() / m
            self.loss_n += 1
            t64 = tt.to(self.t_sum)
            self.t_sum += t64
            self.t_sq += t64 * t64
            self.cx.append(x.reshape(1, -1))
            self.ct.append(tt.reshape(1, -1))
            self._cache = None
        V = len(self.nodes)
        gz = [torch.zeros(n.n_out, device=self.device, dtype=self.dtype) for n in self.nodes]
        gz[-1] = dy.clone()
        by_dst: dict[int, list[Edge]] = {}
        for e in self.edges:
            by_dst.setdefault(e.dst, []).append(e)
        for v in range(V - 1, -1, -1):
            node = self.nodes[v]
            g = gz[v]
            bp = g.abs().mean() if node.n_out else 0.0
            dx = node.store.backward_adam(g, self.lr, self.col_lr_cached(v), self.training)
            if self.training:
                node.ep_bp = node.ep_bp + bp
            for e in reversed(by_dst.get(v, [])):
                if not e.pairs:
                    if self.training:
                        e.ep_out = e.ep_out + bp
                    continue
                ks, js = self._idx(e, self.device)
                d = dx.index_select(0, js)
                if e.src != X:
                    gz[e.src].index_add_(0, ks, e.c * d)
                if self.training:
                    e.ep_out = e.ep_out + bp
                    e.ep_grad = e.ep_grad + d.abs().mean()
        self.step += 1

    # ---------------------------------------------------------- invariants

    def check(self) -> tuple[bool, str]:
        """tnng_check."""
        if not self.nodes:
            return False, "no nodes"
        if self.nodes[-1].n_out != self.n_out:
            return False, "readout width"
        if sum(n.probe for n in self.nodes) > 1:
            return False, "more than one depth dummy"
        for v, n in enumerate(self.nodes):
            if n.n_out == 0:
                return False, f"node {v} has no outputs"
        for i, e in enumerate(self.edges):
            if i and self.edges[i - 1].key() >= e.key():
                return False, f"edges not sorted/unique at {i}"
            if e.dst >= len(self.nodes):
                return False, f"edge {i} dst"
            if e.src != X and e.src >= e.dst:
                return False, f"edge {i} not forward"
            if e.src == X and self.in_degree(e.dst) != 1:
                return False, f"node {e.dst} reads x and more"
            w = self.src_width(e.src)
            for k, j in e.pairs:
                if k >= w or j >= self.nodes[e.dst].n_in:
                    return False, f"edge {i} pair out of range"
            if e.c != 0.0 and abs(e.c - e.in_w * e.out_w) > 1e-12:
                if _rate(e.in_acc, e.acc_n) * _rate(e.out_acc, e.acc_n) != 0.0:
                    return False, f"edge {i} gain != in_w*out_w"
        for v, n in enumerate(self.nodes):
            seen = [0] * n.n_in
            si, deg = 0.0, 0
            for e in self.edges:
                if e.dst != v:
                    continue
                deg += 1
                si += e.in_w
                for _, j in e.pairs:
                    seen[j] += 1
            if any(s != 1 for s in seen):
                return False, f"node {v} columns not written exactly once"
            if deg and abs(si - 1.0) > 1e-9:
                return False, f"node {v} in-weights sum {si}"
        for src in [X, *range(len(self.nodes))]:
            outs = [e.out_w for e in self.edges if e.src == src]
            if outs and abs(sum(outs) - 1.0) > 1e-9:
                return False, f"source {src} out-weights sum {sum(outs)}"
        return True, ""

    def structure(self) -> dict:
        return {"nodes": [[n.store.n_or(k) for k in range(n.n_out)] for n in self.nodes],
                "edges": [(e.src, e.dst, len(e.pairs)) for e in self.edges]}


def _scatter_cols(cat: torch.Tensor, cols: list[int], n_in: int) -> torch.Tensor:
    out = cat.new_zeros(cat.shape[0], n_in)
    idx = torch.tensor(cols, device=cat.device)
    return out.index_copy(1, idx, cat)
