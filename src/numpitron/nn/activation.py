import numpy as np

from numpitron.nn.core import Layer


class ReLU(Layer):
    def __init__(self, name: str = "ReLU"):
        super().__init__(name=name, dtype=None)

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        return {"inputs": np.copy(inputs)}, np.maximum(0, inputs)

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        d_out = (ctx["inputs"] > 0) * d_out
        return {}, d_out
