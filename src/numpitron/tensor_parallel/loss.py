import numpy as np

from numpitron.nn.core import Layer
from numpitron import distributed as npdist


class TensorParallelSoftmaxCrossEntropy(Layer):
    """A Tensor Parallel implementation of the softmax cross-entropy loss.

    Calculation of the loss is done per process per chunk of the vocab embedding
    table. This approach ensures that we send a minimal number of bytes
    over the communication stream.
    """

    def __init__(
        self, name: str = "TensorParallelSoftmaxCrossEntropy", dtype=np.float32
    ):
        super().__init__(name=name, dtype=dtype)

    def forward(
        self,
        params: dict[str, np.ndarray],
        inputs: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        """Calculate the cross-entropy loss given logits (`inputs`).

        Arguments:
            params (dict[str, np.ndarray]): Parameters (if required).
            inputs (B, S, V): A batch (B) of sequences S with vocab size V.
            labels (B, S, 1): A dense set of labels for each batch & sequence.

        Returns:
            Cross-entropy loss.
        """
        batch_size, seq_len, vocab_chunk_size = inputs.shape

        logits = inputs.reshape(batch_size * seq_len, vocab_chunk_size)
        labels = labels.reshape(batch_size * seq_len)

        # Mask labels not in the current logits chunk.
        chunk_start = npdist.tensor_parallel_rank() * vocab_chunk_size
        chunk_end = chunk_start + vocab_chunk_size
        mask = np.logical_or(labels < chunk_start, labels >= chunk_end)
        labels = labels - chunk_start
        labels[mask] = 0

        # Subtract max for stability - dont use keepdims for (B, S) comm.
        logits_max = logits.max(axis=-1)
        npdist.all_reduce(logits_max, op="max", group=npdist.tensor_parallel_group())
        logits = logits - np.expand_dims(logits_max, -1)

        # Mask out the predicted logits where labels were masked, (B, S) comm.
        predicted_logits = logits[np.arange(batch_size * seq_len), labels]
        predicted_logits[mask] = 0.0
        npdist.all_reduce(predicted_logits, group=npdist.tensor_parallel_group())

        # Calculate log-sum-exp, we will need to communicate the sum (B, S).
        exp_logits = np.exp(logits)
        sum_exp_logits = np.sum(exp_logits, axis=-1)
        npdist.all_reduce(sum_exp_logits, group=npdist.tensor_parallel_group())

        loss = -predicted_logits + np.log(sum_exp_logits)
        loss = loss.reshape(batch_size, seq_len)

        ctx = {
            "softmax": (exp_logits / np.expand_dims(sum_exp_logits, -1)).reshape(
                batch_size, seq_len, vocab_chunk_size
            ),
            "mask": mask,
            "masked_labels": labels,
        }

        return ctx, loss

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        """Backwards pass of the softmax-ce loss. All we need to do here
        is to ensure the mask is used accordingly."""
        batch_size, seq_len, vocab_chunk_size = ctx["softmax"].shape

        softmax = ctx["softmax"].reshape(batch_size * seq_len, vocab_chunk_size)
        softmax[np.arange(batch_size * seq_len), ctx["masked_labels"]] -= (
            1 - ctx["mask"]
        )

        return {}, softmax.reshape(batch_size, seq_len, vocab_chunk_size)
