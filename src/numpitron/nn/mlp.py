import numpy as np
from numpitron.nn.activation import ReLU
from numpitron.nn.linear import Linear
from numpitron.nn.model import Model

import numpitron.distributed as npdist


class MLP(Model):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        **kwargs,
    ):
        super().__init__()
        self.add_settings({"d_in": d_in, "d_hidden": d_hidden, "d_out": d_out})
        self.add_layer(
            "column_linear",
            Linear(
                d_in=d_in,
                d_out=d_hidden,
                weight_shard_axis=1,
                bias_shard_axis=0,
                **kwargs,
            ),
        )
        self.add_layer(
            "row_linear",
            Linear(d_in=d_hidden, d_out=d_out, weight_shard_axis=0, **kwargs),
        )
        self.add_layer("relu", ReLU())

        self.row_linear.use_bias = npdist.tensor_parallel_rank() == 0

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = self.column_linear(inputs)
        outputs = self.relu(outputs)
        outputs = self.row_linear(outputs)

        if self.is_scattered and npdist.tensor_parallel_size() > 1:
            npdist.all_reduce(outputs, group=npdist.tensor_parallel_group())

        return outputs

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        d_out = self.row_linear.backward(d_out)
        d_out = self.relu.backward(d_out)
        d_out = self.column_linear.backward(d_out)

        if self.is_scattered and npdist.tensor_parallel_size() > 1:
            npdist.all_reduce(d_out, group=npdist.tensor_parallel_group())

        return d_out
