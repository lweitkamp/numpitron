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
            "embedding": rng.random((self.d_model, self.vocab_size)) * 0.02,
        }
        return {key: value.astype(self.dtype) for key, value in params.items()}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Given an embedding table and input tokens, embed the tokens.

        Arguments:
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        inputs_embedding = np.take(params["weight"].T, inputs, axis=0)
        ctx = {"inputs": inputs, "weight": params["weight"]}
        return inputs_embedding, ctx

    def backward(self, d_out: np.ndarray, ctx: dict) -> tuple[np.ndarray, dict]:
        gradients = {"weight": np.zeros_like(ctx["weight"])}
        np.add.at(gradients["weight"].T, ctx["inputs"], d_out)
        return d_out, gradients


class OutputEmbedding(Layer):
    """The output embedding producing logits. weight are tied with that
    of the input embedding layer."""

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Calculate the logits through a simple matrix product."""
        ctx = {"inputs": inputs, "weight": params["weight"]}
        outputs_embedding = inputs @ params["weight"]
        return outputs_embedding, ctx

    def backward(self, d_out: np.ndarray, ctx: dict) -> tuple[np.ndarray, dict]:
        """Perform a backward pass, calculating the gradients."""
        gradients = {"weight": np.einsum("bsd, bsv -> dv", self.ctx["inputs"], d_out)}
        d_out = d_out @ ctx["weight"].T
        return d_out, gradients


class PositionalEmbedding(Layer):
    """Technically an encoding, just using fourier features."""

    def __init__(
        self, d_model: int, seq_len: int, name="PositionalEmbedding", dtype=np.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.d_model = d_model
        self.seq_len = seq_len

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        pos = np.expand_dims(np.arange(0, self.seq_len), -1)
        _2i = np.arange(self.d_model, step=2) / self.d_model

        encoding = np.zeros((self.seq_len, self.d_model), dtype=self.dtype)
        encoding[:, 0::2] = np.sin(pos / (10000**_2i))
        encoding[:, 1::2] = np.cos(pos / (10000**_2i))

        params: dict[str, np.ndarray] = {"encoding": encoding}
        return {key: value.astype(self.dtype) for key, value in params.items()}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        _, seq_len, *_ = inputs.shape
        inputs_encoding = params["encoding"][:seq_len, :] + inputs
        return inputs_encoding, {}

    def backward(self, d_out: np.ndarray, ctx: dict) -> tuple[np.ndarray, dict]:
        return d_out, {}
