import argparse
from pathlib import Path
import numpy as np

from tqdm import tqdm

from numpitron.optimizer import Adam
from numpitron import distributed as npdist
from numpitron.nn import Transformer, softmax_cross_entropy
from numpitron.data import Tokenizer, DataLoader


def train_step(
    transformer: Transformer, optimizer: Adam, x: np.ndarray, y: np.ndarray
) -> float:
    y_hat = transformer(x)
    loss, d_out = softmax_cross_entropy(y_hat, y)
    transformer.backward(d_out)
    optimizer.step()
    return loss.mean()


def validation_step(transformer: Transformer, x: np.ndarray, y: np.ndarray) -> float:
    y_hat = transformer(x)
    loss, _ = softmax_cross_entropy(y_hat, y)
    return loss.mean()


def checkpoint(model, optimizer, epoch, loss, save_path):
    model.all_gather()
    optimizer.all_gather()
    state = {
        "model_parameters": model.to_dict(),
        "optimizer_parameters": optimizer.to_dict(),
        "epoch": epoch,
        "validation_loss": loss,
    }
    np.save(save_path, state, allow_pickle=True)
    model.scatter()
    optimizer.scatter()


def train(arguments: argparse.Namespace):
    npdist.init(tp_size=arguments.tensor_parallel_size)
    rank = npdist.world_rank()

    tokenizer = Tokenizer.from_pretrained(arguments.tokenizer_path)

    rng = np.random.default_rng(arguments.seed)

    transformer = Transformer(
        d_model=arguments.d_model,
        n_heads=arguments.n_heads,
        n_layers=arguments.n_layers,
        vocab_size=tokenizer.vocab_size,
        seq_len=arguments.sequence_length,
        rng=rng,
    )

    optimizer = Adam(
        model=transformer,
        learning_rate=arguments.learning_rate,
        beta1=arguments.beta1,
        beta2=arguments.beta2,
    )

    train_dataloader = DataLoader(
        arguments.train_dataset_path,
        arguments.sequence_length,
        arguments.batch_size,
        rng,
    )
    validation_dataloader = DataLoader(
        arguments.validation_dataset_path,
        arguments.sequence_length,
        arguments.batch_size,
        rng,
    )

    start_epoch = 0
    min_loss = float("inf")

    if arguments.model_save_path.exists():
        state = np.load(arguments.model_save_path, allow_pickle=True)[()]
        transformer = Transformer.from_dict(state["model_parameters"])
        optimizer = Adam.from_dict(state["optimizer_parameters"], transformer)
        start_epoch = state["epoch"]
        min_loss = state["validation_loss"]

    transformer.scatter()
    optimizer.scatter()

    for epoch in tqdm(range(start_epoch, arguments.n_epochs), disable=rank != 0):

        ### Training Step ###
        train_bar = train_dataloader.iter_epoch()
        for x, y in train_bar:
            checkpoint(
                transformer,
                optimizer,
                epoch,
                0,
                arguments.model_save_path,
            )

            train_loss = train_step(transformer, optimizer, x, y)
            train_bar.set_description(f"Train (loss: {train_loss:.3f})")
            train_bar.refresh()

        ### Validation Step ###
        validation_bar = validation_dataloader.iter_epoch()
        for x, y in validation_bar:
            validation_loss = validation_step(transformer, x, y)

        ### Save State ###
        if validation_loss < min_loss:
            min_loss = validation_loss
            checkpoint(
                transformer,
                optimizer,
                epoch,
                validation_loss,
                arguments.model_save_path,
            )


parser = argparse.ArgumentParser("Transformer model Trainer")

# Transformer related.
parser.add_argument("--d-model", type=int, default=128)
parser.add_argument("--sequence-length", type=int, default=64)
parser.add_argument("--n-heads", type=int, default=8)
parser.add_argument("--n-layers", type=int, default=4)

# Data related
parser.add_argument("--batch-size", type=int, default=12)
parser.add_argument("--n-epochs", type=int, default=100)
parser.add_argument(
    "--tokenizer-path", type=Path, default="examples/shakespeare_tokenizer.json"
)
parser.add_argument(
    "--train-dataset-path", type=Path, default="examples/shakespeare_char_train.bin"
)
parser.add_argument(
    "--validation-dataset-path", type=Path, default="examples/shakespeare_char_val.bin"
)
parser.add_argument("--model-save-path", type=Path, default="examples/model.npy")
parser.add_argument("--seed", type=int, default=42)

# Optimization related.
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.99)

# Parallelization related.
parser.add_argument("--tensor-parallel-size", type=int, default=1)


if __name__ == "__main__":
    train(parser.parse_args())
