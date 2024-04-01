import argparse
from pathlib import Path
import json

from numpy.random import Generator
from tqdm import trange

import numpy as np
from numpitron import nn, models, data
from numpitron.nn.core import Sequential


def generate(
    state: models.State,
    model: Sequential,
    tokenizer: data.Tokenizer,
    num_samples: int,
    rng: Generator,
    seq_len: int,
    initial_prompt: str = "\n",
    temperature: float = 1.0,
    top_k: int = 5,
) -> str:
    """Sample tokens from a model."""
    predicted_text = ""
    tokens = tokenizer.encode(initial_prompt)

    for _ in trange(num_samples):
        tokens = tokens[-seq_len:]
        _, logits = model(state.parameters, np.asarray([tokens]))
        probabilities = nn.softmax(logits[0, -1] / temperature)

        next_token = np.argmax(rng.multinomial(n=1, pvals=probabilities))
        tokens.append(next_token)

        predicted_text += tokenizer.decode([next_token])

    return f"\033[1m{initial_prompt}\033[0m{predicted_text}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-path",
        type=Path,
        default="examples/shakespeare_Transformer.npy",
        help="Path to numpy pickled state file.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="examples/shakespeare_transformer.json",
        help="Path to json config file.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default="examples/shakespeare_tokenizer.json",
        help="Path to JSON tokenizer file.",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default="\n",
        help="Starting prompt.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=250,
        help="Number of tokens to sample."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Control the generation temperature."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Sample only from top-K most likely tokens."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed."
    )
    args = parser.parse_args()

    with open(args.config_path, mode="r", encoding="utf-8") as f:
        cfg = json.load(f)

    output = generate(
        state=models.State.from_pretrained(args.state_path),
        model=models.from_config(cfg),
        tokenizer=data.Tokenizer.from_pretrained(args.tokenizer_path),
        num_samples=args.num_samples,
        rng=np.random.default_rng(args.seed),
        seq_len=cfg["model_config"]["seq_len"],
        initial_prompt=args.initial_prompt,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(output)
