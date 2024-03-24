import numpy as np
from numpy.random import Generator

from numpitron import nn
from numpitron.nn.core import Layer, Sequential


class TransformerBlock(Layer):
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
            self.d_model, self.n_heads, self.d_model // self.n_heads
        )
        self.norm1 = nn.LayerNorm(self.d_model)
        self.mlp = nn.MLP(self.d_model, self.d_model * 4)
        self.norm2 = nn.LayerNorm(self.d_model)

    def init_params(
        self, rng: Generator
    ) -> dict[
        str,
        np.ndarray,
    ]:
        params: dict[
            str,
            np.ndarray,
        ] = {
            "attention": self.attention.init_params(rng),
            "norm1": self.norm1.init_params(rng),
            "mlp": self.mlp.init_params(rng),
            "norm2": self.norm2.init_params(rng),
        }
        return params

    def forward(
        self,
        params: dict[
            str,
            np.ndarray,
        ],
        inputs: np.ndarray,
    ) -> tuple[
        dict,
        np.ndarray,
    ]:
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

    def backward(
        self,
        ctx: dict,
        d_out: np.ndarray,
    ) -> tuple[
        dict,
        np.ndarray,
    ]:
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
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        name: str = "Transformer",
        dtype=np.float32,
    ):
        super().__init__(name=name, dtype=dtype)

        self.layers.append(nn.InputEmbedding(d_model, vocab_size))
        self.layers.append(nn.PositionalEmbedding(d_model, seq_len))
        self.layers.extend(
            [
                TransformerBlock(d_model, n_heads, name=f"TransformerBlock_{i}")
                for i in range(n_layers)
            ]
        )
        self.layers.append(nn.OutputEmbedding(d_model, vocab_size))

    def init_params(
        self, rng: Generator
    ) -> dict[
        str,
        np.ndarray,
    ]:
        params = super().init_params(rng)
        params[self.layers[-1].name] = params[self.layers[0].name]
        return params

    def forward(
        self,
        params: dict[str, np.ndarray],
        inputs: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        ce = nn.SoftmaxCrossEntropy()

        ctx, logits = super().forward(params, inputs)
        ctx_cross_entorpy, loss = ce.forward(params, logits, labels)

        ctx[ce.name] = ctx_cross_entorpy
        return ctx, loss

    def backward(self, ctx: dict) -> tuple[dict, np.ndarray]:
        ce = nn.SoftmaxCrossEntropy()
        _, d_out = ce.backward(ctx[ce.name], None)
        gradients, d_out = super().backward(ctx, d_out)
        return gradients
