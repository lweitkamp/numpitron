import argparse
from pathlib import Path

import numpy as np
from numpy.random import Generator
from tqdm import trange

from numpitron import distributed as npdist
from numpitron.data import Tokenizer
from numpitron.nn import Transformer, softmax


def basic_sample(logits, temperature: float, rng: Generator, vocab_size: int) -> int:
    """Sampling according to a softmax distribution with temperature, but
    we will materialize the entire logits tensor on device (inefficient)."""
    assert logits.shape[0] == 1, "Only single batch size supported"
    logits = logits[0, -1, :]
    logits_full = np.empty(vocab_size, dtype=np.float32)

    npdist.all_gather(logits, logits_full, group=npdist.tensor_parallel_group())

    probabilities = softmax((logits_full / temperature).astype(np.float64))
    token = rng.multinomial(1, probabilities).argmax()
    return token


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


def generate(arguments: argparse.Namespace):
    npdist.init(tp_size=arguments.tensor_parallel_size)

    state = np.load(arguments.model_save_path, allow_pickle=True)[()]
    transformer = Transformer.from_dict(state["model_parameters"])
    tokenizer = Tokenizer.from_pretrained(arguments.tokenizer_path)

    transformer.scatter()

    print(f"Loading model trained for {state['epoch']} epochs with val loss {state['validation_loss']}")

    rng = np.random.default_rng(arguments.seed)

    predicted_text = ""
    tokens = tokenizer.encode(arguments.initial_prompt)

    for _ in trange(arguments.num_samples):
        tokens = tokens[-transformer.seq_len :]
        logits = transformer(np.asarray([tokens]))

        next_token = basic_sample(logits, arguments.temperature, rng, tokenizer.vocab_size)
        tokens.append(next_token)

        predicted_text += tokenizer.decode([next_token])

    if npdist.tensor_parallel_rank() == 0:
        print(f"\033[1m{arguments.initial_prompt}\033[0m{predicted_text}")

parser = argparse.ArgumentParser()

parser.add_argument("--model-save-path", type=Path, default="examples/model.npy")
parser.add_argument(
    "--tokenizer-path", type=Path, default="examples/shakespeare_tokenizer.json"
)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--initial-prompt", type=str, default="\n")
parser.add_argument("--num-samples", type=int, default=250)
parser.add_argument("--temperature", type=float, default=1.0)

parser.add_argument("--tensor-parallel-size", type=int, default=1)

if __name__ == "__main__":
    generate(parser.parse_args())
