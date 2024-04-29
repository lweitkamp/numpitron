import numpy as np

from numpitron.nn.layer import Layer


class LayerNorm(Layer):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.add_settings({"d_model": d_model, "eps": eps})
        self.add_parameter("weight", np.ones((d_model,)))
        self.add_parameter("bias", np.ones((d_model,)))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        mean = inputs.mean(axis=-1, keepdims=True)
        var = inputs.var(axis=-1, keepdims=True)
        normed = (inputs - mean) / np.sqrt(var + self.eps)
        outputs = self.weight.data * normed + self.bias.data

        self.ctx |= {"inputs": inputs, "mean": mean, "var": var}
        return outputs

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        inputs, mean, var = (
            self.ctx.pop("inputs"),
            self.ctx.pop("mean"),
            self.ctx.pop("var"),
        )

        inputs_normed = (inputs - mean) / np.sqrt(var + self.eps)

        self.update_parameter(
            "weight", gradient=np.sum(d_out * inputs_normed, axis=(0, 1))
        )
        self.update_parameter("bias", gradient=d_out.sum(axis=(0, 1)))

        wdy = self.weight.data * d_out
        c1 = np.sum(inputs_normed * wdy, axis=-1) / self.d_model
        c2 = wdy.sum(axis=-1) / self.d_model
        d_out = (wdy - c1[..., None] * inputs_normed - c2[..., None]) / inputs.std(
            axis=-1, keepdims=True
        )

        return d_out
