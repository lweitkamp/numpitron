import numpy as np
from numpitron.nn.core import Layer, weight_init_fn


class Linear(Layer):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        use_bias: bool = True,
        weight_shard_axis: int | None = None,
        bias_shard_axis: int | None = None,
        weight_init: str = "scaled_normal",
        bias_init: str = "scaled_normal",
        **kwargs,
    ):
        super().__init__()
        self.add_settings(
            {
                "d_in": d_in,
                "d_out": d_out,
                "use_bias": use_bias,
                "weight_shard_axis": weight_shard_axis,
                "bias_shard_axis": bias_shard_axis,
                "weight_init": weight_init,
                "bias_init": bias_init,
            }
        )

        self.add_parameter(
            "weight",
            weight_init_fn(weight_init, **kwargs)((d_in, d_out)),
            weight_shard_axis,
        )
        if self.use_bias:
            self.add_parameter(
                "bias", weight_init_fn(bias_init, **kwargs)((d_out,)), bias_shard_axis
            )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.einsum("bsd, dh -> bsh", inputs, self.weight.data)

        if self.use_bias:
            outputs = outputs + self.bias.data

        self.ctx["inputs"] = inputs
        return outputs

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        inputs = self.ctx.pop("inputs")

        self.update_parameter(
            "weight", gradient=np.einsum("bsd, bsh -> dh", inputs, d_out)
        )
        if self.use_bias:
            self.update_parameter("bias", gradient=d_out.sum(axis=(0, 1)))

        return np.einsum("bsh, dh -> bsd", d_out, self.weight.data)
