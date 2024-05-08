import argparse
from pathlib import Path

from tqdm import trange

import numpy as np

from numpitron.nn import Transformer, softmax
from numpitron.data import Tokenizer


def generate(arguments: argparse.Namespace):
    state = np.load(arguments.model_save_path, allow_pickle=True)[()]
    transformer = Transformer.from_dict(state["model_parameters"])
    tokenizer = Tokenizer.from_pretrained(arguments.tokenizer_path)
    print(f"Loading model trained for {state['epoch']} epochs with val loss {state['validation_loss']}")

    rng = np.random.default_rng(arguments.seed)

    predicted_text = ""
    tokens = tokenizer.encode(arguments.initial_prompt)

    for _ in trange(arguments.num_samples):
        tokens = tokens[-transformer.seq_len :]
        logits = transformer(np.asarray([tokens]))
        probabilities = softmax(
            (logits[0, -1] / arguments.temperature).astype(np.float64)
        )

        next_token = np.argmax(rng.multinomial(n=1, pvals=probabilities))
        tokens.append(next_token)

        predicted_text += tokenizer.decode([next_token])

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

if __name__ == "__main__":
    generate(parser.parse_args())
