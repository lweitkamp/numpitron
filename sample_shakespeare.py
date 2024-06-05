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

    tokens = np.array([tokenizer.encode("\n")] * arguments.num_samples, dtype=np.uint16)
    if npdist.data_parallel_size() > 1:
        scattered_tokens = np.empty(
            (arguments.num_samples // npdist.data_parallel_size(), 1),
            dtype=np.uint16,
        )
        npdist.scatter(
            tokens, scattered_tokens, axis=0, group=npdist.data_parallel_group()
        )
        tokens = scattered_tokens

    for _ in trange(transformer.seq_len - 1):
        logits = transformer(tokens)

        if arguments.sampler == "softmax":
            next_tokens = softmax_sample(
                logits, arguments.temperature, rng, tokenizer.vocab_size
            )
        elif arguments.sampler == "greedy":
            next_tokens = greedy_sample(logits)

        tokens = np.concatenate([tokens, np.array(next_tokens)[:, None]], axis=-1)

    tokens = tokens.astype(np.uint16)
    if npdist.data_parallel_size() > 1:
        gathered_tokens = np.zeros((arguments.num_samples, transformer.seq_len), dtype=np.uint16)
        npdist.all_gather(tokens, gathered_tokens, axis=0, group=npdist.data_parallel_group())
        tokens = gathered_tokens

    if npdist.world_rank() == 0:
        for i, batch in enumerate(map(tokenizer.decode, tokens)):
            print(f"Batch {i}:{batch}\n")


parser = argparse.ArgumentParser()

parser.add_argument("--model-save-path", type=Path, default="data/model.npy")
parser.add_argument(
    "--tokenizer-path", type=Path, default="data/shakespeare_tokenizer.json"
)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--sampler", type=str, default="softmax")
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1.0)

parser.add_argument("--tensor-parallel-size", type=int, default=1)
parser.add_argument("--data-parallel-size", type=int, default=1)

if __name__ == "__main__":
    generate(parser.parse_args())
