import numpy as np
from numpy.random import Generator

import numpitron.nn as nn
from numpitron.nn.layer import Layer


class Attention(Layer):
    """A Multi-headed self-Attention (decoder-only) layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
    ):
        super().__init__()
        self.add
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.scale = np.sqrt(self.d_hidden)

        self.qkv = nn.Linear(
            self.d_model, (self.n_heads, self.d_hidden), "qkv", self.dtype
        )
        self.out = nn.Linear(
            (self.n_heads, self.d_hidden), self.d_model, "out", self.dtype
        )

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {
            "q_projection": self.qkv.init_params(rng),
            "k_projection": self.qkv.init_params(rng),
            "v_projection": self.qkv.init_params(rng),
            "out_projection": self.out.init_params(rng),
        }
        return params

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        """Forward pass through the self-attention layer."""
        seq_len = inputs.shape[1]
        mask = np.expand_dims(np.tri(seq_len, seq_len, dtype=bool), (0, 1))

        q_ctx, q = self.qkv(params["q_projection"], inputs)
        k_ctx, k = self.qkv(params["k_projection"], inputs)
        v_ctx, v = self.qkv(params["v_projection"], inputs)

        attention_weights = np.einsum("bshm, bzhm -> bhsz", q, k) / self.scale
        attention_weights = np.where(mask, attention_weights, float("-inf"))
        softmax_ctx, attention_weights = nn.Softmax(axis=-1)({}, attention_weights)

        attention = np.einsum("bhsz, bzhm -> bshm", attention_weights, v)
        out_projection_ctx, out = self.out(params["out_projection"], attention)

        ctx = {
            "inputs": np.copy(inputs),
            "attention_weights": np.copy(attention_weights),
            "mask": np.copy(mask),
            "q": np.copy(q),
            "k": np.copy(k),
            "v": np.copy(v),
            "softmax_ctx": softmax_ctx,
            "q_projection_ctx": q_ctx,
            "k_projection_ctx": k_ctx,
            "v_projection_ctx": v_ctx,
            "out_projection_ctx": out_projection_ctx,
        }

        return ctx, out

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        """Backward pass through the Attention layer."""

        out_projection_gradients, d_out = self.out.backward(
            ctx["out_projection_ctx"], d_out
        )

        d_out_v = np.matmul(
            ctx["attention_weights"].transpose(0, 1, 3, 2),
            d_out.transpose(0, 2, 1, 3),
        ).transpose(0, 2, 1, 3)

        d_out = np.matmul(d_out.transpose(0, 2, 1, 3), ctx["v"].transpose(0, 2, 3, 1))

        _, d_out = nn.Softmax(axis=-1).backward(ctx["softmax_ctx"], d_out)
        d_out = np.where(ctx["mask"], d_out, 0) / self.scale

        d_out_q = np.matmul(d_out, ctx["k"].transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        d_out_k = np.einsum("bhsz, bshm -> bzhm", d_out, ctx["q"])

        q_projection_gradients, d_out_q = self.qkv.backward(
            ctx["q_projection_ctx"], d_out_q
        )
        k_projection_gradients, d_out_k = self.qkv.backward(
            ctx["k_projection_ctx"], d_out_k
        )
        v_projection_gradients, d_out_v = self.qkv.backward(
            ctx["v_projection_ctx"], d_out_v
        )

        d_out = d_out_q + d_out_k + d_out_v

        gradients = {
            "q_projection": q_projection_gradients,
            "k_projection": k_projection_gradients,
            "v_projection": v_projection_gradients,
            "out_projection": out_projection_gradients,
        }

        return gradients, d_out
