import numpy as np

from numpitron import nn
from numpitron.nn.core import Sequential


class MLP(Sequential):
    """Simple Multi-Layered Perceptron with two layers."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        use_bias: bool = True,
        name="MLP",
        dtype=np.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.d_model = d_model
        self.d_hidden = d_hidden

        self.layers.extend(
            [
                nn.Linear(d_model, d_hidden, use_bias, "Linear1", dtype),
                nn.ReLU(),
                nn.Linear(d_hidden, d_model, use_bias, "Linear2", dtype),
            ]
        )
