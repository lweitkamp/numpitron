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

    dataset_train_path: Path
    dataset_validation_path: Path

    save_path: Path

    from_savefile: bool = False
    seed: int = 42


def load_config(config_path: Path | str) -> Config:
    config = json.load(Path(config_path).open(mode="r"))
    cfg = Config(**config)

    cfg.save_path = Path(cfg.save_path)
    cfg.dataset_train_path = Path(cfg.dataset_train_path)
    cfg.dataset_validation_path = Path(cfg.dataset_validation_path)
    return cfg
