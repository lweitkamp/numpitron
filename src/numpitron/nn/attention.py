import numpy as np
from numpitron.nn.activation import Softmax
from numpitron.nn.linear import Linear
from numpitron.nn.model import Model

import numpitron.distributed as npdist


class Attention(Model):
    def __init__(self, d_model: int, n_heads: int, d_hidden: int, **kwargs):
        super().__init__()
        qkv_projection = Linear(
            d_model,
            n_heads * d_hidden * 3,
            weight_shard_axis=1,
            use_bias=False,
            **kwargs,
        )
        out_projection = Linear(
            n_heads * d_hidden,
            d_model,
            weight_shard_axis=0,
            use_bias=False,
            **kwargs,
        )
        self.add_layer("qkv_projection", qkv_projection)
        self.add_layer("out_projection", out_projection)
        self.add_layer("softmax", Softmax(axis=-1))

        self.add_settings({"d_mdel": d_model, "n_heads": n_heads, "d_hidden": d_hidden})

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = inputs.shape

        qkv = self.qkv_projection(inputs)
        qkv = qkv.reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        q, k, v = np.split(qkv, 3, -1)

        mask = np.expand_dims(np.tri(seq_len, seq_len, dtype=bool), (0, 1))
        attention_weights = np.matmul(q, np.swapaxes(k, -2, -1)) / (d_model**0.5)
        attention_weights = np.where(mask, attention_weights, float("-inf"))

        attention_weights = self.softmax(attention_weights)
        attention = np.matmul(attention_weights, v)

        attention = attention.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        outputs = self.out_projection(attention)

        self.ctx |= {
            "mask": mask,
            "q": q,
            "k": k,
            "v": v,
            "attention_weights": attention_weights,
        }

        if self.is_scattered and npdist.tensor_parallel_size() > 1:
            print("HAHHAHA")
            npdist.all_reduce(outputs)

        return outputs

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = d_out.shape

        d_out = self.out_projection.backward(d_out)
        d_out = d_out.reshape(batch_size, seq_len, self.n_heads, -1).transpose(
            0, 2, 1, 3
        )

        d_out_v = np.matmul(
            self.ctx.pop("attention_weights").transpose(0, 1, 3, 2), d_out
        )
        d_out = np.matmul(d_out, self.ctx.pop("v").transpose(0, 1, 3, 2))

        d_out = self.softmax.backward(d_out)
        d_out = np.where(self.ctx.pop("mask"), d_out, 0) / (d_model**0.5)

        d_out_q = np.matmul(d_out, self.ctx.pop("k"))
        d_out_k = np.matmul(self.ctx.pop("q").transpose(0, 1, 3, 2), d_out).transpose(
            0, 1, 3, 2
        )

        d_out = (
            np.concatenate([d_out_q, d_out_k, d_out_v], axis=-1)
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, seq_len, -1)
        )
        d_out = self.qkv_projection.backward(d_out)

        if self.is_scattered and npdist.tensor_parallel_size() > 1:
            npdist.all_reduce(d_out)

        return d_out
