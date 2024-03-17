import numpy as np
from numpy.random import Generator

from numpitron.nn.core import Layer


class LayerNorm(Layer):
    """Layer normalization - normalize the inputs over the last dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        name: str = "Linear",
        dtype=np.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.d_model = d_model
        self.eps = eps

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {
            "weight": rng.random((self.d_model,)) * 0.02,
            "bias": np.zeros((self.d_model,)),
        }
        return {key: value.astype(self.dtype) for key, value in params.items()}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        """Calculate mean and standard deviation of the inputs along the
        last dimension and normalize the inputs. Additionally,
        multiply the normalized input with weights and add a bias."""
        mean = inputs.mean(axis=-1, keepdims=True)
        var = inputs.var(axis=-1, keepdims=True)
        normed = (inputs - mean) / np.sqrt(var + self.eps)
        outputs = params["weight"] * normed + params["bias"]

        ctx = {
            "weight": params["weight"],
            "inputs": inputs,
            "mean": mean,
            "var": var,
        }

        return ctx, outputs

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        """The most straightforward reference is surpisingly from the Triton tutorial
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html."""

        inputs_normed = (ctx["inputs"] - ctx["mean"]) / np.sqrt(ctx["var"] + self.eps)

        gradients = {
            "weight": np.sum(d_out * inputs_normed, axis=(0, 1)),
            "bias": d_out.sum(axis=(0, 1)),
        }

        wdy = ctx["weight"] * d_out
        c1 = np.sum(inputs_normed * wdy, axis=-1) / self.d_model
        c2 = wdy.sum(axis=-1) / self.d_model
        d_out = (wdy - c1[..., None] * inputs_normed - c2[..., None]) / ctx[
            "inputs"
        ].std(axis=-1, keepdims=True)

        return gradients, d_out
