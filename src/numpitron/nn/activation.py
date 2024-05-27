import numpy as np

from numpitron.nn.core import Layer


def softmax(inputs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of x along the given axis."""
    x_ = np.exp(inputs - np.max(inputs, axis=axis, keepdims=True))
    return x_ / x_.sum(axis=axis, keepdims=True)


class Softmax(Layer):
    def __init__(self, axis: int = -1, **kwargs):
        super().__init__()
        self.add_setting("axis", axis)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = softmax(inputs, self.axis)
        self.ctx["outputs"] = outputs
        return outputs

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        outputs = self.ctx.pop("outputs")
        _, _, seq_len, _ = outputs.shape

        left = np.einsum(
            "...ij, jk -> ...ijk", outputs, np.eye(seq_len, dtype=d_out.dtype)
        )
        right = np.einsum("...ij, ...ik -> ...ijk", outputs, outputs)
        d_out = (d_out[..., None, :] @ (left - right)).reshape(outputs.shape)
        return d_out


class ReLU(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.ctx["inputs"] = inputs
        return np.maximum(0, inputs)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return (self.ctx.pop("inputs") > 0) * d_out
