import numpy as np

from numpitron.nn.core import Layer
from numpitron import nn


class SoftmaxCrossEntropy(Layer):
    """Softmax cross-entropy loss function."""

    def __init__(self, name: str = "SoftmaxCrossEntropy", dtype=np.float32):
        super().__init__(name=name, dtype=dtype)

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray, labels: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        """Calculate the cross-entropy loss given logits (`inputs`).

        Arguments:
            params (dict[str, np.ndarray]): Parameters (if required).
            inputs (B, S, V): A batch (B) of sequences S with vocab size V.
            labels (B, S, 1): A dense set of labels for each batch & sequence.

        Returns:
            Cross-entropy loss.
        """
        ctx = {
            "inputs": inputs,
            "labels": labels,
        }

        batch_size, seq_len, vocab_size = inputs.shape

        logits = inputs.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        # Subtract max for stability.
        logits = logits - logits.max(axis=-1, keepdims=True)

        predicted_logits = logits[np.arange(batch_size * seq_len), labels]
        logsumexp_logits = np.log(np.sum(np.exp(logits), axis=-1))

        loss = logsumexp_logits - predicted_logits
        loss = loss.reshape(batch_size, seq_len)

        return ctx, loss

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        """Backwards pass of the softmax-ce loss.
        
        @TODO(laurens): maybe refactor loss functions to a Loss clas or so.
        """
        batch_size, seq_len, vocab_size = ctx["inputs"].shape

        logits = ctx["inputs"].reshape(batch_size * seq_len, vocab_size)
        labels = ctx["labels"].reshape(batch_size * seq_len)

        probabilities = nn.softmax(logits, axis=-1)
        probabilities[np.arange(batch_size * seq_len), labels] -= 1

        return {}, probabilities.reshape(batch_size, seq_len, vocab_size)
