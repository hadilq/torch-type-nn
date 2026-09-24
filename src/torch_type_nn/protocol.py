"""The contract between the structure scaler and a model family.

type-nn's scaling rule is architecture-agnostic: on each axis a *probe* is
an exact identity that back-prop may move; a probe that moved past
``theta(T)`` is promoted (under ``rule="bic"``: and pays its BIC price); late
in training items back inside the identity band are removed (under
``rule="bic"``: items are ablated and removed while the model criterion
allows).
What is specific to an architecture is only *what* an item is: how to
insert an identity, how far an item is from it, how to reset it and how to
remove it. That is this protocol.

Items live on three axes:

``WIDTH``
    a coordinate between two layers (a unit and the weights that read it)
``DEGREE``
    a factor inside a unit (for type-nn: an Or inside an And)
``DEPTH``
    a layer (there is at most one depth probe at a time)

An adapter may leave an axis empty (return no sites or candidates).
:class:`~torch_type_nn.adapters.typenn.TypeNNAdapter` is the reference
implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn

__all__ = ["WIDTH", "DEGREE", "DEPTH", "AXES", "Item", "Trial", "EditContext", "Scalable"]

WIDTH, DEGREE, DEPTH = "width", "degree", "depth"
AXES = (WIDTH, DEGREE, DEPTH)


@dataclass(frozen=True)
class Item:
    """One structural item: an axis and an adapter-defined address.

    Addresses are only valid until the next structural edit.
    """

    axis: str
    address: tuple


@dataclass
class Trial:
    """A structural edit made tentatively: keep it with ``commit`` or revert it with ``undo``."""

    undo: Callable[[], None]
    commit: Callable[[], None]


@dataclass
class EditContext:
    """What a new probe needs: its birth step, noise amplitude, and the RNG."""

    step: int
    noise: float
    generator: torch.Generator


@runtime_checkable
class Scalable(Protocol):
    """A model family the :class:`~torch_type_nn.scaling.StructureScaler` can grow and prune."""

    model: nn.Module

    # ------------------------------------------------------------ bookkeeping
    def num_params(self) -> int:
        """Every learnable scalar currently present (probes included)."""

    def set_tracking(self, on: bool) -> None:
        """Collect (or stop collecting) training statistics in forward passes."""

    def reset_stats(self) -> None:
        """Forget the statistics of the epoch that ended."""

    def set_edit_sink(self, sink: list | None) -> None:
        """Report parameter replacements (``Edit``) to ``sink`` while it is a list."""

    def finalize(self) -> None:
        """Training is over; tidy storage (e.g. drop free padding)."""

    # ----------------------------------------------------------------- probes
    def probe_sites(self, axis: str) -> Sequence[Hashable]:
        """Where probes of ``axis`` live now (WIDTH and DEGREE), in visiting order."""

    def probe_at(self, axis: str, site: Hashable | None = None) -> Item | None:
        """The probe at ``site`` (for DEPTH: the depth probe, ``site`` ignored)."""

    def add_probe(self, axis: str, site: Hashable, ctx: EditContext) -> None:
        """Insert an exact identity (up to ``ctx.noise``) at ``site``."""

    def promote(self, item: Item) -> None:
        """The probe becomes a live item."""

    def born(self, item: Item) -> int:
        """Training step at which ``item`` was inserted."""

    def displacement(self, item: Item) -> float:
        """RMS distance of ``item`` from its identity (``inf`` if it has none)."""

    def best_site(self, axis: str, moving: Item | None = None) -> Hashable:
        """Where a new probe of ``axis`` is most wanted (DEPTH: the loudest gap,
        counted as if ``moving`` were absent)."""

    def site_of(self, item: Item) -> Hashable:
        """The site a probe occupies, in :meth:`best_site` coordinates."""

    def remove_probe(self, item: Item) -> None:
        """Take a probe out (near-exactly) so it can move elsewhere."""

    # -------------------------------------------------- measurement / pruning
    def ablate(self, item: Item) -> Callable[[], None] | None:
        """Reset ``item`` to its identity in place; return the exact undo
        (``None`` if the item has no identity)."""

    def cost(self, item: Item) -> int:
        """Parameters owned by ``item``."""

    def trial_remove(self, item: Item) -> Trial | None:
        """Remove ``item`` (DEPTH) tentatively; ``None`` if it cannot be removed."""

    def removal_order(self, axis: str) -> Sequence[Item]:
        """Live items of ``axis`` to try removing one at a time (DEPTH)."""

    def prune_candidates(self) -> Sequence[Item]:
        """Items (WIDTH and DEGREE) for the joint ablation pool, in a fixed order."""

    def can_remove(self, item: Item, removed: Sequence[Item]) -> bool:
        """Whether ``item`` may join the ablated set ``removed`` (keep-one rules)."""

    def params_without(self, removed: Sequence[Item]) -> int:
        """Exact parameter count of the model with ``removed`` taken out."""

    def commit_removals(self, removed: Sequence[Item],
                        axes: Sequence[str] = (WIDTH, DEGREE)) -> dict[str, int]:
        """Remove ``removed`` for real and clear the surviving probe flags, on the
        given ``axes`` only (the threshold rule commits width, then degree).
        Returns counter increments (``or_drop``, ``and_add``, ...)."""
