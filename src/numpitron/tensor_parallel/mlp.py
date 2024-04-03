import numpy as np

from numpitron import nn
from numpitron.nn.core import Sequential, tree_map

from numpitron import distributed as npdist


class TensorParallelMLP(Sequential):
    """Simple Multi-Layered Perceptron with two layers."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        communicator: npdist.ParallelCommunicator,
        name="TensorParallelMLP",
        dtype=np.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.comm = communicator

        self.layers.append(
            nn.Linear(d_model, d_hidden // self.comm.tp_size, "ColumnLinear", dtype)
        )
        self.layers.append(nn.ReLU())
        self.layers.append(
            nn.Linear(d_hidden // self.comm.tp_size, d_model, "RowLinear", dtype),
        )

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        ctxs, inputs = super().forward(params, inputs)
        tree_map(npdist.all_reduce, inputs, comm=self.comm.tp_comm)
        return ctxs, inputs

    def backward(
        self,
        ctx: dict[str, dict],
        d_out: np.ndarray,
    ) -> tuple[dict[str, dict], np.ndarray]:
        gradients, d_out = super().backward(ctx, d_out)
        npdist.all_reduce(d_out, comm=self.comm.tp_comm)
        return gradients, d_out
