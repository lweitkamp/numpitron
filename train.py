"""A simple training script."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpitron import data, models, nn, optimizers, distributed as npdist
from tqdm import tqdm, trange


def step(
    state: models.State,
    model: nn.core.Sequential,
    optimizer: optimizers.Adam,
    loss: nn.core.Sequential,
    inputs: np.ndarray,
    labels: np.ndarray,
) -> models.State:
    """Perform a forward + backward pass through the model and update
    the parameters using Adam."""
    # Forward pass.
    ctx, logits = model(state.parameters, inputs)
    ctx_loss, loss_value = loss(state.parameters, logits, labels)

    # Backward pass.
    _, d_out = loss.backward(ctx_loss, None)
    gradients, d_out = model.backward(ctx, d_out)

    # Update optimizer and get a new state.
    new_optimizer_state, new_parameters = optimizer.step(
        state.optimizer_state,
        gradients,
        state.parameters,
    )

    return replace(
        state,
        parameters=new_parameters,
        optimizer_state=new_optimizer_state,
        final_loss=loss_value,
    )


def validation_step(model, parameters, pbar):
    """An iteration of the validation dataset."""
    validation_loss = []
    for _, (inputs, labels) in pbar:
        _, loss = model(parameters, inputs, labels)
        validation_loss.append(loss)
    return np.mean(validation_loss)


def train(config: dict, save_path: str | Path) -> dict:
    save_path = Path(save_path)

    rng = np.random.default_rng(config["seed"])

    model = models.from_config(config)
    parameters = model.init_params(rng)
    optimizer = optimizers.from_config(config)
    optimizer_state = optimizer.init_state(parameters)
    loss_fn = nn.from_config(config["loss"])
    state = models.State(parameters, optimizer_state)

    train_dataloader, validation_dataloader = data.get_dataloader(config, rng)

    min_loss = float("inf")

    for epoch in trange(
        config["data_config"]["num_epochs"],
        desc="Epochs",
        disable=npdist.world_rank() != 0,
    ):
        train_bar = tqdm(
            enumerate(train_dataloader.iter_epoch()),
            leave=False,
            desc="Train (loss: N/A)",
            total=train_dataloader.batches_per_epoch,
            disable=npdist.world_rank() != 0,
        )

        for i, (inputs, labels) in train_bar:
            state = step(state, model, optimizer, loss_fn, inputs, labels)
            loss = state.final_loss.mean()

            train_bar.set_description(f"Train (loss: {loss:.3f})")
            train_bar.refresh()

            with (save_path / "log.csv").open(mode="a", encoding="utf-8") as f:
                f.write(f"{epoch},{i},{loss}\n")

            if loss < min_loss:
                state.save(save_path / f"shakespeare_{model.__class__.__name__}.npy")

        if epoch % 5 == 0:
            validation_bar = tqdm(
                enumerate(validation_dataloader.iter_epoch()),
                leave=False,
                desc="Validation (loss: N/A)",
                total=validation_dataloader.batches_per_epoch,
                disable=npdist.world_rank() != 0,
            )
            validation_loss = validation_step(
                state.model, state.parameters, validation_bar
            )
            print(f"{epoch} : validation loss - {validation_loss:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="examples/shakespeare_transformer.json",
        help="Path to json config file.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="examples",
        help="Where to save outputs.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of tensor parallel processes.",
    )
    args = parser.parse_args()

    with open(args.config_path, mode="r", encoding="utf-8") as f:
        cfg = json.load(f)

    npdist.init(
        tp_size=args.tensor_parallel_size,
    )

    train(cfg, args.save_path)
