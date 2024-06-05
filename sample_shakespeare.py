import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

from numpitron import distributed as npdist
from numpitron.data import Tokenizer
from numpitron.nn import Transformer
from numpitron.inference import greedy_sample, softmax_sample


def generate(arguments: argparse.Namespace):
    npdist.init(
        tp_size=arguments.tensor_parallel_size, dp_size=arguments.data_parallel_size
    )

    state = np.load(arguments.model_save_path, allow_pickle=True)[()]
    transformer = Transformer.from_dict(state["model_parameters"])
    tokenizer = Tokenizer.from_pretrained(arguments.tokenizer_path)

    transformer.scatter()

    print(
        f"Loading model trained for {state['epoch']} epochs with val loss {state['validation_loss']}"
    )

    rng = np.random.default_rng(arguments.seed)

    predicted_text = ""
    tokens = np.array(
        [tokenizer.encode(arguments.initial_prompt)] * arguments.num_samples
    )

    # TODO Scatter tokens if data parallel is enabled
    if npdist.data_parallel_size() > 1:
        pass

    for _ in trange(arguments.sample_length):
        tokens = tokens[-transformer.seq_len :]
        logits = transformer(np.asarray([tokens]))

        if arguments.sampler == "softmax":
            next_token = softmax_sample(
                logits, arguments.temperature, rng, tokenizer.vocab_size
            )
        elif arguments.sampler == "greedy":
            next_token = greedy_sample(logits)

        tokens.append(next_token)

        predicted_text += tokenizer.decode([next_token])

    # TODO gather tokens if data parallel is enabled
    if npdist.data_parallel_size() > 1:
        pass

    # TODO print for each batch
    if npdist.tensor_parallel_rank() == 0:
        for batch_idx in arguments.num_samples:
            print(f"\033[1m{arguments.initial_prompt}\033[0m{predicted_text}")


parser = argparse.ArgumentParser()

parser.add_argument("--model-save-path", type=Path, default="data/model.npy")
parser.add_argument(
    "--tokenizer-path", type=Path, default="data/shakespeare_tokenizer.json"
)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--sampler", type=str, default="softmax")
parser.add_argument("--initial-prompt", type=str, default="\n")
parser.add_argument("--sample-length", type=int, default=250)
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1.0)

parser.add_argument("--tensor-parallel-size", type=int, default=1)
parser.add_argument("--data-parallel-size", type=int, default=1)

if __name__ == "__main__":
    generate(parser.parse_args())
