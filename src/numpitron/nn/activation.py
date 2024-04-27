import numpy as np

from numpitron.nn.layer import Layer


def softmax(inputs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of x along the given axis."""
    x_ = np.exp(inputs - np.max(inputs, axis=axis, keepdims=True))
    return x_ / x_.sum(axis=axis, keepdims=True)


# class Softmax(Layer):
#     """Softmax layer. Has no associated weights."""

#     def __init__(self, axis: int = -1, name: str = "Softmax", dtype=None):
#         super().__init__(name=name, dtype=dtype)
#         self.axis = axis

#     def forward(
#         self, params: dict[str, np.ndarray], inputs: np.ndarray
#     ) -> tuple[dict, np.ndarray]:
#         outputs = softmax(inputs)
#         return {"inputs_sm": np.copy(outputs)}, outputs

#     def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
#         inputs_sm = ctx["inputs_sm"]
#         _, _, seq_len, _ = inputs_sm.shape

#         left = np.einsum("...ij, jk -> ...ijk", inputs_sm, np.eye(seq_len))
#         right = np.einsum("...ij, ...ik -> ...ijk", inputs_sm, inputs_sm)
#         d_out = (d_out[..., None, :] @ (left - right)).reshape(inputs_sm.shape)
#         return {}, d_out


class ReLU(Layer):
    """ReLU non-linear activation function. Has no associated weights."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.ctx["inputs"] = inputs
        return np.maximum(0, inputs)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return (self.ctx.pop("inputs") > 0) * d_out
