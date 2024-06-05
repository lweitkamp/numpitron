from functools import partial
from typing import Self

import numpy as np

from numpitron import distributed as npdist
from numpitron.nn import (
    InputEmbedding,
    OutputEmbedding,
    PositionalEncoding,
    TransformerBlock,
)
from numpitron.nn.model import Model

npdist.init(tp_size=npdist.world_size())


class Transformer(Model):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        seq_len: int,
        **kwargs,
    ):
        super().__init__()
        self.add_settings(
            {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "vocab_size": vocab_size,
                "seq_len": seq_len,
            }
        )
        block = partial(TransformerBlock, d_model, n_heads, **kwargs)

        self.add_layer("input_embedding", InputEmbedding(d_model, vocab_size, **kwargs))
        self.add_layer(
            "positional_encoding", PositionalEncoding(d_model, seq_len, **kwargs)
        )
        for layer in range(n_layers):
            self.add_layer(f"block_{layer}", block())
        self.add_layer("output_embedding", OutputEmbedding(self.input_embedding))

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        for layer in list(self.layers.values())[::-1]:
            d_out = layer.backward(d_out)

            if npdist.data_parallel_size() > 1:
                for parameter in layer.parameters.values():
                    # Inplace allreduce on frozen dict object is a bit sketchy.
                    npdist.all_reduce(
                        parameter.gradient, group=npdist.data_parallel_group()
                    )

        return d_out

    @classmethod
    def from_dict(cls, model_dict: dict[str, dict]) -> Self:
        transformer = super().from_dict(model_dict)

        # Tie embeddings.
        transformer.output_embedding.input_embedding = transformer.input_embedding
        return transformer
