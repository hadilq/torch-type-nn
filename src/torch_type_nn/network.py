"""A stack of type-nn layers whose depth, widths and degrees can change."""

from __future__ import annotations

import math
import re

import torch
from torch import nn

from .layer import AndOr

__all__ = ["TypeNN", "birth_depth"]


def birth_depth(in_features: int, out_features: int) -> int:
    """round(ln(1 + n m)), at least 1: xor 1, iris 3, wine 4, wdbc 3, diabetes 2, ionosphere 4."""
    # C lround: half away from zero (Python round() is half-to-even)
    return max(1, int(math.floor(math.log(1.0 + in_features * out_features) + 0.5)))


class TypeNN(nn.Module):
    """A dense stack of :class:`AndOr` layers, ``n -> m -> ... -> m``.

    At birth every unit is a single random Or (a degree-1 And) and there are
    :func:`birth_depth` layers. Nothing about the structure is a
    hyper-parameter: train it with :class:`~torch_type_nn.scaling.StructureScaler`
    (or :func:`~torch_type_nn.train.fit`) and width, degree and depth are
    grown early and pruned late. Without a scaler it is an ordinary static
    network.

    Args:
        in_features, out_features: task widths (fixed for life).
        depth: birth depth; default :func:`birth_depth`.
        seed: seeds ``self.generator``, used for the birth draw and for
            every random draw of structural edits.
    """

    def __init__(self, in_features: int, out_features: int, *, depth: int | None = None,
                 seed: int = 1, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.generator = torch.Generator().manual_seed(int(seed))
        D = birth_depth(in_features, out_features) if depth is None else int(depth)
        self.birth_depth = D
        layers, n = [], self.in_features
        for _ in range(D):
            layers.append(AndOr(n, self.out_features, 1, generator=self.generator,
                                device=device, dtype=dtype))
            n = self.out_features
        self.layers = nn.ModuleList(layers)

    @property
    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def num_params(self) -> int:
        """Every learnable scalar present: (n_in + 2) per live Or, probes included."""
        return sum(layer.num_params() for layer in self.layers)

    def insert_layer(self, index: int, layer: AndOr) -> None:
        self.layers.insert(index, layer)

    def remove_layer(self, index: int) -> AndOr:
        layer = self.layers[index]
        del self.layers[index]
        return layer

    def structure(self) -> list[list[int]]:
        """Live Ors per unit, per layer."""
        return [layer.num_ors().tolist() for layer in self.layers]

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"depth={self.depth}, params={self.num_params()}")

    def get_extra_state(self):
        return {"birth_depth": self.birth_depth, "generator": self.generator.get_state()}

    def set_extra_state(self, state) -> None:
        self.birth_depth = int(state.get("birth_depth", self.birth_depth))
        if "generator" in state:
            self.generator.set_state(state["generator"])

    # torch.Generator does not pickle: carry its state instead
    def __getstate__(self):
        state = self.__dict__.copy()
        gen = state.pop("generator")
        state["_generator_state"] = gen.get_state()
        return state

    def __setstate__(self, state):
        gen_state = state.pop("_generator_state", None)
        self.__dict__.update(state)
        self.generator = torch.Generator()
        if gen_state is not None:
            self.generator.set_state(gen_state)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accepts a checkpoint of any structure: depth is taken from the keys,
        widths and degrees from the tensors."""
        idx = {int(m.group(1)) for k in state_dict
               if (m := re.match(r"layers\.(\d+)\.weight$", k))}
        D = max(idx) + 1 if idx else 0
        ref = self.layers[0].weight if len(self.layers) else torch.empty(0)
        while len(self.layers) < D:
            self.layers.append(AndOr(1, 1, 1, device=ref.device, dtype=ref.dtype))
        while len(self.layers) > D:
            del self.layers[len(self.layers) - 1]
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
