import numpy as np
from numpy.random import Generator

from numpitron.nn.core import Layer


class InputEmbedding(Layer):
    """The input embedding lookup-table."""

    def __init__(
        self, d_model: int, vocab_size: int, name="InputEmbedding", dtype=np.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {
            "embedding": rng.random((self.d_model, self.vocab_size)),
        }
        return {key: value.astype(self.dtype) for key, value in params.items()}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        """Given an embedding table and input tokens, embed the tokens.

        Arguments:
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        inputs_embedding = np.take(params["embedding"].T, inputs, axis=0)
        ctx = {"inputs": inputs, "embedding": params["embedding"]}
        return ctx, inputs_embedding

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[np.ndarray, dict]:
        gradients = {"embedding": np.zeros_like(ctx["embedding"])}
        np.add.at(gradients["embedding"].T, ctx["inputs"], d_out)
        return gradients, d_out


class OutputEmbedding(Layer):
    """The output embedding producing logits. weight are tied with that
    of the input embedding layer."""

    def __init__(
        self, d_model: int, vocab_size: int, name="OutputEmbedding", dtype=np.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        return {}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        """Calculate the logits through a simple matrix product."""
        ctx = {"inputs": inputs, "embedding": params["embedding"]}
        outputs_embedding = inputs @ params["embedding"]
        return ctx, outputs_embedding

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        """Perform a backward pass, calculating the gradients."""
        gradients = {"embedding": np.einsum("bsd, bsv -> dv", ctx["inputs"], d_out)}
        d_out = d_out @ ctx["embedding"].T
        return gradients, d_out


class PositionalEmbedding(Layer):
    """Technically an encoding, just using fourier features."""

    def __init__(
        self, d_model: int, seq_len: int, name="PositionalEmbedding", dtype=np.float32
    ):
        super().__init__(name=name, dtype=dtype)
        pos = np.expand_dims(np.arange(0, seq_len), -1)
        _2i = np.arange(d_model, step=2) / d_model
        encoding = np.zeros((seq_len, d_model), dtype=dtype)
        encoding[:, 0::2] = np.sin(pos / (10000**_2i))
        encoding[:, 1::2] = np.cos(pos / (10000**_2i))
        self.encoding = encoding

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        _, seq_len, *_ = inputs.shape
        inputs_encoding = self.encoding[:seq_len, :] + inputs
        return {}, inputs_encoding

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        return {}, d_out
