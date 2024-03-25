"""A simple training script."""
import argparse

import numpy as np
from numpy.random import Generator
from tqdm import tqdm, trange

from numpitron import save_params, load_params
from numpitron.model import Config, load_config, Transformer
from numpitron.optimizer import Adam
from numpitron.data import DataLoader


def init_model(config: Config, rng: Generator):
    """Initialize the model and parameters, load from saved if possible."""
    model = Transformer(
        config.vocab_size,
        config.seq_len,
        config.d_model,
        config.n_heads,
        config.n_layers,
    )
    if config.from_savefile:
        parameters = load_params(config.save_path / "parameters.npy")
        parameters[model.layers[-1].name] = parameters[model.layers[0].name]
    else:
        parameters = model.init_params(rng)
    return model, parameters


def init_optimizer(config: Config, parameters: dict):
    """Initialize the optimizer and state, load from saved if possible."""
    optimizer = Adam(config.learning_rate, config.betas)
    if config.from_savefile:
        state = load_params(config.save_path / "optimizer.npy")
    else:
        state = optimizer.init_state(parameters)
    return optimizer, state


def init_dataloader(config: Config, rng: Generator):
    """Create the train and validation data loaders."""
    train_dataloader = DataLoader(
        dataset_path=config.dataset_train_path,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        rng=rng,
    )
    validation_dataloader = DataLoader(
        dataset_path=config.dataset_validation_path,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        rng=rng,
    )
    return train_dataloader, validation_dataloader


def train_step(model, parameters, optimizer, state, inputs, labels):
    """A single forward pass, calculating the gradients & updating parameters."""
    ctx, loss = model(parameters, inputs, labels)
    gradients = model.backward(ctx)
    state, parameters = optimizer.step(state, gradients, parameters)
    return state, parameters, loss


def validation_step(model, parameters, pbar):
    """An iteration of the validation dataset."""
    validation_loss = []
    for _, (inputs, labels) in pbar:
        _, loss = model(parameters, inputs, labels)
        validation_loss.append(loss)
    return np.mean(validation_loss)


def train(config: Config) -> dict:
    rng = np.random.default_rng(config.seed)

    model, parameters = init_model(config, np.random.default_rng(555))
    optimizer, state = init_optimizer(config, parameters)
    train_dataloader, validation_dataloader = init_dataloader(config, rng)

    min_loss = float("inf")

    for epoch in trange(config.num_epochs, desc="Epochs"):
        train_bar = tqdm(
            enumerate(train_dataloader.iter_epoch()),
            leave=False,
            desc="Train (loss: N/A)",
            total=train_dataloader.batches_per_epoch,
        )

        for step, (inputs, labels) in train_bar:
            state, parameters, train_loss = train_step(
                model, parameters, optimizer, state, inputs, labels
            )

            train_bar.set_description(f"Train (loss: {train_loss.mean():.3f})")
            train_bar.refresh()

            with (config.save_path / "log.csv").open(mode="a", encoding="utf-8") as f:
                f.write(f"{epoch},{step},{train_loss.mean()}\n")

            if train_loss.mean() < min_loss:
                save_params(config.save_path / "parameters.npy", parameters)
                save_params(config.save_path / "optimizer.npy", state)

        if epoch % 5 == 0:
            validation_bar = tqdm(
                enumerate(validation_dataloader.iter_epoch()),
                leave=False,
                desc="Validation (loss: N/A)",
                total=validation_dataloader.batches_per_epoch,
            )
            validation_loss = validation_step(model, parameters, validation_bar)
            print(f"{epoch} : validation loss - {validation_loss:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="examples/transformer.json",
        help="Path to json config file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config_path)
    train(cfg)
