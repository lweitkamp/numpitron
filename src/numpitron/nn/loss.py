import numpy as np

from numpitron import distributed as npdist


def softmax_cross_entropy(inputs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch_size, seq_len, vocab_chunk_size = inputs.shape

    logits = inputs.reshape(batch_size * seq_len, vocab_chunk_size)
    labels = labels.reshape(batch_size * seq_len)

    chunk_start = npdist.get_var("embedding_chunk_start")
    chunk_end = npdist.get_var("embedding_chunk_end")

    # Mask labels not in the current logits chunk.
    mask = np.logical_or(labels < chunk_start, labels >= chunk_end)
    labels = labels - chunk_start
    labels[mask] = 0

    # Subtract max for stability - dont use keepdims for (B, S) comm.
    logits_max = logits.max(axis=-1)

    if npdist.tensor_parallel_size() > 1:
        npdist.all_reduce(
            logits_max, op="max", group=npdist.tensor_parallel_group()
        )

    logits = logits - np.expand_dims(logits_max, -1)

    # Mask out the predicted logits where labels were masked, (B, S) comm.
    predicted_logits = logits[np.arange(batch_size * seq_len), labels]
    predicted_logits[mask] = 0.0

    if npdist.tensor_parallel_size() > 1:
        npdist.all_reduce(predicted_logits, group=npdist.tensor_parallel_group())

    # Calculate log-sum-exp, we will need to communicate the sum (B, S).
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=-1)

    if npdist.tensor_parallel_size() > 1:
        npdist.all_reduce(sum_exp_logits, group=npdist.tensor_parallel_group())

    loss = -predicted_logits + np.log(sum_exp_logits)
    loss = loss.reshape(batch_size, seq_len)

    softmax = (exp_logits / np.expand_dims(sum_exp_logits, -1))
    softmax[np.arange(batch_size * seq_len), labels] -= 1 - mask

    return loss, softmax.reshape(batch_size, seq_len, vocab_chunk_size)
