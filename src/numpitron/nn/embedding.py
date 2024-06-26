from typing import Self

import numpy as np

import numpitron.distributed as npdist
from numpitron.nn.core import Layer, weight_init_fn


class InputEmbedding(Layer):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        weight_init: str = "scaled_normal",
        **kwargs,
    ):
        super().__init__()
        self.add_settings({"d_model": d_model, "vocab_size": vocab_size})
        self.add_parameter(
            "embedding",
            weight_init_fn(weight_init, **kwargs)((d_model, vocab_size)),
            shard_axis=1,
        )
        npdist.set_var("embedding_chunk_start", 0)
        npdist.set_var("embedding_chunk_end", self.embedding.data.shape[1] - 1)

    def scatter(self, src: int = 0):
        total_vocab_size = self.embedding.data.shape[1]

        super().scatter(src)
        
        if npdist.tensor_parallel_size() > 1:
            chunk_size = total_vocab_size // npdist.tensor_parallel_size()
            chunks = np.arange(0, total_vocab_size, chunk_size)
            chunks[1:] += total_vocab_size % npdist.tensor_parallel_size()
            npdist.set_var("embedding_chunk_start", chunks[npdist.tensor_parallel_rank()])
            npdist.set_var("embedding_chunk_end", chunks[npdist.tensor_parallel_rank() + 1])

    def all_gather(self):
        super().all_gather()
        npdist.set_var("embedding_chunk_start", 0)
        npdist.set_var("embedding_chunk_end", self.embedding.data.shape[1] - 1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        chunk_start = npdist.get_var("embedding_chunk_start")
        chunk_end = npdist.get_var("embedding_chunk_end")
        
        mask = np.logical_or(inputs < chunk_start, inputs >= chunk_end)

        # Set tokens to chunk range, mask tokens outside range.
        if self.is_scattered:
            inputs = inputs - chunk_start
            inputs[mask] = 0

        # Take the correct embeddings and mask outside range.
        inputs_embedding = np.take(self.embedding.data.T, inputs, axis=0)

        if self.is_scattered:
            inputs_embedding[mask, :] = 0.0
            
        if self.is_scattered and npdist.tensor_parallel_size() > 1:
            npdist.all_reduce(inputs_embedding, group=npdist.tensor_parallel_group())

        self.ctx["inputs"] = inputs
        self.ctx["mask"] = mask

        return inputs_embedding

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        gradient = (
            np.zeros_like(self.embedding.data)
            if self.embedding.gradient is None
            else self.embedding.gradient
        )
        inputs = self.ctx.pop("inputs")
        mask = self.ctx.pop("mask")
        if self.is_scattered:
            np.add.at(gradient.T, inputs[~mask], d_out[~mask])
        else:
            np.add.at(gradient.T, inputs, d_out)

        self.update_parameter("embedding", gradient=gradient)
        return None


class OutputEmbedding(Layer):
    def __init__(self, input_embedding: Layer, **kwargs):
        super().__init__()
        self.input_embedding = input_embedding

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs_embedding = inputs @ self.input_embedding.embedding.data
        self.ctx["inputs"] = inputs
        return outputs_embedding

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        self.input_embedding.update_parameter(
            "embedding",
            gradient=np.einsum("bsd, bsv -> dv", self.ctx.pop("inputs"), d_out),
        )

        d_out = d_out @ self.input_embedding.embedding.data.T

        if self.is_scattered and npdist.tensor_parallel_size() > 1:
            npdist.all_reduce(d_out, group=npdist.tensor_parallel_group())

        return d_out

    @classmethod
    def from_dict(cls, layer_dict: dict[str, dict]) -> Self:
        return cls(None)


class PositionalEncoding(Layer):
    def __init__(self, d_model: int, seq_len: int, **kwargs):
        super().__init__()
        self.add_settings({"d_model": d_model, "seq_len": seq_len})
        pos = np.expand_dims(np.arange(0, seq_len), -1)
        _2i = np.arange(d_model, step=2) / d_model
        encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        encoding[:, 0::2] = np.sin(pos / (10000**_2i))
        encoding[:, 1::2] = np.cos(pos / (10000**_2i))
        self.encoding = encoding

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        _, seq_len, *_ = inputs.shape
        inputs_encoding = self.encoding[:seq_len, :] + inputs
        return inputs_encoding
