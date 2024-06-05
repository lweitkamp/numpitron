from numpy.random import Generator
import numpy as np

import numpitron.distributed as npdist
from numpitron.nn import softmax


def greedy_sample(logits) -> int:
    """Greedy sampling - just an argmax."""
    assert logits.shape[0] == 1, "Only single batch size supported"
    logits = logits[0, -1, :]

    rank_max_index = logits.argmax()
    rank_max_value = logits[rank_max_index]

    # Communicate both the argmax index (but then normalized with location)
    # and the max value. return the index of the max value.
    rank_max_index += npdist.tensor_parallel_rank() * len(logits)

    max_indices = np.zeros(npdist.tensor_parallel_size(), dtype=np.float32)
    max_values = np.zeros(npdist.tensor_parallel_size(), dtype=np.float32)

    max_values[npdist.tensor_parallel_rank()] = rank_max_value
    max_indices[npdist.tensor_parallel_rank()] = rank_max_index

    npdist.all_reduce(max_values, group=npdist.tensor_parallel_group())
    npdist.all_reduce(max_indices, group=npdist.tensor_parallel_group())

    return int(max_indices[max_values.argmax()])


def softmax_sample(logits, temperature: float, rng: Generator, vocab_size: int) -> int:
    """Sampling according to a softmax distribution with temperature, but
    we will materialize the entire logits tensor on device (inefficient)."""
    assert logits.shape[0] == 1, "Only single batch size supported"
    logits = logits[0, -1, :]
    logits_full = np.empty(vocab_size, dtype=np.float32)

    npdist.all_gather(logits, logits_full, group=npdist.tensor_parallel_group())

    probabilities = softmax((logits_full / temperature).astype(np.float64))
    token = rng.multinomial(1, probabilities).argmax()
    return token
