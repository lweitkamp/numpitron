import string

import numpy as np
from numpy.random import Generator

from numpitron.nn.core import Layer


def to_tuple(dim: tuple | int) -> tuple[int, ...]:
    return tuple([dim]) if isinstance(dim, int) else dim


class Linear(Layer):
    """A generalized linear layer. Works on integer and tuple dimensions."""

    def __init__(
        self,
        input_dim: tuple | int,
        output_dim: tuple | int,
        use_bias:  bool = True,
        name: str = "Linear",
        dtype=np.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        input_dim = to_tuple(input_dim)
        output_dim = to_tuple(output_dim)

        # format the einsums for this layer.
        ascii_options = list(string.ascii_letters)
        self.in_chr = "".join(ascii_options.pop() for _ in range(len(input_dim)))
        self.out_chr = "".join(ascii_options.pop() for _ in range(len(output_dim)))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {
            "weight": rng.random(self.input_dim + self.output_dim) * 0.02,
        }
        if self.use_bias:
            params["bias"] = np.zeros(self.output_dim)
        return {key: value.astype(self.dtype) for key, value in params.items()}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        """Perform a forward pass in the linear layer, y = Wx + b."""
        ctx: dict = {"inputs": np.copy(inputs), "weight": params["weight"]}

        outputs = np.einsum(
            f"...{self.in_chr}, ...{self.in_chr}{self.out_chr} -> ...{self.out_chr}",
            inputs,
            params["weight"],
        )

        if self.use_bias:
            outputs = outputs + params["bias"]

        return ctx, outputs

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        """Perform a backward pass, calculating the gradients."""
        weight_gradient = np.einsum(
            f"...{self.in_chr}, ...{self.out_chr} -> ...{self.in_chr}{self.out_chr}",
            ctx["inputs"],
            d_out,
        )

        gradients = {"weight": weight_gradient.sum(axis=(0, 1))}
        if self.use_bias:
            gradients["bias"] = d_out.sum(axis=(0, 1))


        d_out = np.einsum(
            f"...{self.out_chr}, {self.in_chr}{self.out_chr} -> ...{self.in_chr}",
            d_out,
            ctx["weight"],
        )
        return gradients, d_out
