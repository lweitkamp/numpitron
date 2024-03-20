import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class Config:
    batch_size: int
    num_epochs: int

    vocab_size: int

    seq_len: int
    d_model: int
    n_heads: int
    n_layers: int

    learning_rate: float
    betas: tuple[float, float]

    dataset_train_path: str
    dataset_validation_path: str

    seed: int = 42


def load_config(config_path: Path | str) -> Config:
    config = json.load(Path(config_path).open(mode="r"))
    return Config(**config)
