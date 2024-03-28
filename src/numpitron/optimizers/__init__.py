# flake8: noqa
from numpitron.optimizers.adam import Adam


def from_config(cfg: dict):
    """Return a model class from a config file."""
    optimizer_class = {"Adam": Adam}[cfg["optimizer_type"]]

    optimizer = optimizer_class(
        **cfg["optimizer_config"]
    )
    return optimizer
