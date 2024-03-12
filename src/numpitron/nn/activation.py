import numpy as np

from numpitron.nn.core import Layer


class ReLU(Layer):
    def __init__(self, name: str = "ReLU"):
        super().__init__(name=name, dtype=None)

    def forward(self, params, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs), {"inputs": np.copy(inputs)}

    def backward(self, d_out: np.ndarray, ctx: dict) -> tuple[np.ndarray, dict]:
        d_out = (ctx["inputs"] > 0) * d_out
        return d_out, {}
