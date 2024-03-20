import argparse
from pathlib import Path

import numpy as np
from numpy.random import Generator

from numpitron import models, nn, load_params


def sample(
    model,
    params,
    initial_prompt: str,
    meta: dict,
    max_len: int,
    seq_len: int,
    rng: Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    predicted_text = ""
    tokens = [meta["stoi"][x] for x in initial_prompt]

    for _ in range(max_len):
        tokens = tokens[-seq_len:]
        _, logits = model(params, np.asarray([tokens]))
        probabilities = nn.softmax(logits[0, -1] / temperature)

        next_token = np.argmax(rng.multinomial(n=1, pvals=probabilities))
        tokens.append(next_token)

        predicted_text += meta["itos"][next_token]

    print(f"\033[1m{initial_prompt}\033[0m{predicted_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=Path,
        default="examples/transformer.json",
        help="Path to json config file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default="examples/transformer.npy",
        help="Path to numpy pickled model file.",
    )
    parser.add_argument(
        "--vocab-meta-path",
        type=Path,
        default="examples/meta.pkl",
        help="Path to metadata for vocab.",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default="\n",
        help="Starting prompt.",
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=500,
        help="Number of tokens to sample."
    )
    args = parser.parse_args()

    config = models.load_config(args.config_path)

    transformer = models.Transformer(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dtype=np.float32,
    )
    params = load_params(args.model_path)

    vocab_meta = np.load(args.vocab_meta_path, allow_pickle=True)

    sample(
        transformer,
        params,
        args.initial_prompt,
        vocab_meta,
        args.sample_length,
        config.seq_len,
        rng=np.random.default_rng(config.seed),
        temperature=1.0,
        top_k=5,
    )
