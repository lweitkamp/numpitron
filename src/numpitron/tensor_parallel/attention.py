import numpy as np

from numpitron import nn
from numpitron.nn.core import tree_map

from numpitron import distributed as npdist


class TensorParallelAttention(nn.Attention):
    """A Multi-headed self-Attention (decoder-only) layer that is split
    on the head-dimension."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        communicator: npdist.ParallelCommunicator,
        name: str = "TensorParallelAttention",
        dtype=np.float32,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads // communicator.tp_size,
            d_hidden=d_hidden,
            name=name,
            dtype=dtype,
        )
        self.comm = communicator

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        """Forward pass through the self-attention layer."""
        ctx, out = super().forward(params, inputs)
        tree_map(npdist.all_reduce, out, comm=self.comm.tp_comm)
        return ctx, out

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        """Backward pass through the Attention layer."""
        gradients, d_out = super().backward(ctx, d_out)
        tree_map(npdist.all_reduce, gradients, comm=self.comm.tp_comm)
        return gradients, d_out
