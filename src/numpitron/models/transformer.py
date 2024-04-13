import numpy as np
from numpy.random import Generator

from numpitron import nn
from numpitron.nn.core import Layer, Sequential


class TransformerBlock(Layer):
    """Simple transformer block with
    attn -> norm -> res -> mlp -> norm -> res.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        name: str = "TransformerBlock",
        dtype=np.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.d_model = d_model
        self.n_heads = n_heads

        self.attention = nn.Attention(
            self.d_model,
            self.n_heads,
            self.d_model // self.n_heads,
            dtype=dtype,
        )
        self.norm1 = nn.LayerNorm(self.d_model, dtype=dtype)
        self.mlp = nn.MLP(self.d_model, self.d_model * 4, dtype=dtype)
        self.norm2 = nn.LayerNorm(self.d_model, dtype=dtype)

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        params = {
            "attention": self.attention.init_params(rng),
            "norm1": self.norm1.init_params(rng),
            "mlp": self.mlp.init_params(rng),
            "norm2": self.norm2.init_params(rng),
        }
        return params

    def forward(
        self,
        params: dict[str, np.ndarray],
        inputs: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        attention_ctx, inputs_ = self.attention(params["attention"], inputs)
        norm1_ctx, inputs_ = self.norm1(params["norm1"], inputs_)
        inputs = inputs_ + inputs

        mlp_ctx, inputs_ = self.mlp(params["mlp"], inputs)
        norm2_ctx, inputs_ = self.norm1(params["norm2"], inputs_)
        inputs = inputs_ + inputs

        ctx = {
            "attention": attention_ctx,
            "norm1": norm1_ctx,
            "mlp": mlp_ctx,
            "norm2": norm2_ctx,
        }

        return ctx, inputs

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        norm2_gradients, d_out = self.norm2.backward(ctx["norm2"], d_out)
        mlp_gradients, d_out = self.mlp.backward(ctx["mlp"], d_out)
        norm1_gradients, d_out = self.norm2.backward(ctx["norm1"], d_out)
        attention_gradients, d_out = self.attention.backward(ctx["attention"], d_out)

        gradients = {
            "attention": attention_gradients,
            "norm1": norm1_gradients,
            "mlp": mlp_gradients,
            "norm2": norm2_gradients,
        }

        return gradients, d_out


class Transformer(Sequential):
    """Transformer model."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        input_embedding: Layer = nn.InputEmbedding,
        transformer_block: Layer = TransformerBlock,
        name: str = "Transformer",
        dtype=np.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        block_name = transformer_block.__class__.__name__

        self.layers.append(input_embedding(d_model, vocab_size, dtype=dtype))
        self.layers.append(nn.PositionalEmbedding(d_model, seq_len, dtype=dtype))
        self.layers.extend(
            [
                transformer_block(
                    d_model, n_heads, name=f"{block_name}_{i}", dtype=dtype
                )
                for i in range(n_layers)
            ]
        )
        self.layers.append(nn.OutputEmbedding(d_model, vocab_size, dtype=dtype))

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        params = super().init_params(rng)
        params[self.layers[-1].name] = params[self.layers[0].name]
        return params

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        gradients, d_out = super().backward(ctx, d_out)
        gradients[self.layers[0].name]["embedding"] += gradients["OutputEmbedding"][
            "embedding"
        ]
        return gradients, d_out
