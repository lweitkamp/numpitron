from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class State:
    """Simple model dataclass that holds the 'state' for training. A state
    should be sufficient to reconstruct the (1) model and (2) optimizer class,
    and start training and sampling from the model."""
    parameters: dict
    optimizer_state: dict
    final_loss: float = 0

    @classmethod
    def from_pretrained(cls, path: Path | str):
        """Load a State from a NumPy pickle file."""
        return cls(**np.load(path, allow_pickle=True)[()])

    def save(self, path: Path | str) -> None:
        """Store a State to a NumPy pickle file."""
        obj = {
            "parameters": self.parameters,
            "optimizer_state": self.optimizer_state,
            "final_loss": self.final_loss,
        }
        np.save(path, obj)
