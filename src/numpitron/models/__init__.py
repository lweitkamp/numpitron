# flake8: noqa
from numpitron.models.transformer import Transformer
from numpitron.models.megatron_lm import MegatronLM
from numpitron.models.state import State

from numpitron.nn.core import Sequential


def from_config(cfg: dict) -> Sequential:
    """Return a model class from a config file."""
    model_class = {
        "Transformer": Transformer,
        "Megatron-LM": MegatronLM,
    }[cfg["model_type"]]

    model = model_class(
        **cfg["model_config"]
    )
    return model
