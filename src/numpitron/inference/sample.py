from numpy.random import Generator
import numpy as np

import numpitron.distributed as npdist
from numpitron.nn import softmax


def greedy_sample(logits) -> int:
    """Greedy sampling - just an argmax."""
    logits = logits[:, -1, :]

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
    logits = logits[:, -1, :]

    if npdist.tensor_parallel_size() > 1:
        logits_full = np.zeros(vocab_size, dtype=np.float32)
        npdist.all_gather(logits, logits_full, group=npdist.tensor_parallel_group())
        logits = logits_full

    probabilities = softmax((logits / temperature).astype(np.float64))
    tokens = list(map(lambda probs: rng.multinomial(1, probs).argmax(), probabilities))
    return tokens
