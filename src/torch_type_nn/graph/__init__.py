"""graph-type-nn: type-nn when the layers are the nodes of a DAG.

Three implementations of one rule (type-nn's ``type_nn_graph.c``):

* :class:`GraphTypeNN` + :class:`GraphScaler`: an ``nn.Module`` of
  :class:`~torch_type_nn.layer.AndOr` nodes (batched, autograd, TypeAdam);
* :class:`~torch_type_nn.native.NativeGraphTypeNN`: the C reference itself;
* :class:`~torch_type_nn.cuda.CudaGraphTypeNN`: packed nodes on a device.

The rule is written once (:mod:`.core`); the backends differ only in how a
node stores its Ors.
"""

from .core import Edge, GraphCore, NodeState, NodeStore
from .torch_store import GraphScaler, GraphTypeNN, Node, TorchStore, fit_graph, load_snapshot

__all__ = ["GraphTypeNN", "GraphScaler", "Node", "fit_graph", "load_snapshot",
           "GraphCore", "NodeStore", "NodeState", "Edge", "TorchStore"]
