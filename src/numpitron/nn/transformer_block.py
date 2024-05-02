
import numpy as np

from numpitron.nn import LayerNorm, MLP, Attention
from numpitron.nn.model import Model


class TransformerBlock(Model):
    def __init__(self, d_model: int, n_heads: int, **kwargs):
        super().__init__()
        self.add_settings({"d_model": d_model, "n_heads": n_heads})
        self.add_layer("norm1", LayerNorm(d_model, **kwargs))
        self.add_layer(
            "attention", Attention(d_model, n_heads, d_model // n_heads, **kwargs)
        )
        self.add_layer("norm2", LayerNorm(d_model, **kwargs))
        self.add_layer("mlp", MLP(d_model, d_model * 4, d_model, **kwargs))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = self.norm1(self.attention(inputs)) + inputs
        outputs = self.norm2(self.mlp(outputs)) + outputs
        return outputs

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        d_out = self.mlp.backward(self.norm2.backward(d_out))
        d_out = self.attention.backward(self.norm1.backward(d_out))
        return d_out
