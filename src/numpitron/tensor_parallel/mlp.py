import numpy as np

from numpitron import nn
from numpitron.nn.core import tree_map

from numpitron import distributed as npdist


class TensorParallelMLP(nn.MLP):
    """Simple Multi-Layered Perceptron with two layers."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        name="TensorParallelMLP",
        dtype=np.float32,
    ):
        super().__init__(
            d_model=d_model,
            d_hidden=d_hidden // npdist.tensor_parallel_size(),
            name=name,
            dtype=dtype,
        )

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        ctxs, outputs = super().forward(params, inputs)
        tree_map(npdist.all_reduce, outputs, group=npdist.tensor_parallel_group())
        return ctxs, outputs

    def backward(
        self,
        ctx: dict[str, dict],
        d_out: np.ndarray,
    ) -> tuple[dict[str, dict], np.ndarray]:
        gradients, d_out = super().backward(ctx, d_out)
        npdist.all_reduce(d_out, group=npdist.tensor_parallel_group())
        return gradients, d_out
