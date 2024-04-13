import numpy as np

from numpitron import tensor_parallel as tp
from numpitron.models.transformer import TransformerBlock, Transformer


class MegatronLMBlock(TransformerBlock):
    """Simple transformer block with
    attn -> norm -> res -> mlp -> norm -> res.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        name: str = "MegatronLMBlock",
        dtype=np.float32,
    ):
        super().__init__(d_model=d_model, n_heads=n_heads, name=name, dtype=dtype)
        self.d_model = d_model
        self.n_heads = n_heads

        self.attention = tp.TensorParallelAttention(
            self.d_model,
            self.n_heads,
            self.d_model // self.n_heads,
            dtype=dtype,
        )
        self.mlp = tp.TensorParallelMLP(
            self.d_model,
            self.d_model * 4,
            dtype=dtype,
        )


class MegatronLM(Transformer):
    """Transformer model."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        name: str = "MegatronLM",
        dtype=np.float32,
    ):
        super().__init__(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            input_embedding=tp.TensorParallelInputEmbedding,
            transformer_block=MegatronLMBlock,
            name=name,
            dtype=dtype,
        )
