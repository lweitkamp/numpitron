from typing import Callable
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class Parameter:
    data: np.ndarray
    gradient: np.ndarray
    shard_axis: int | None = None


class Layer:
    def __init__(self):
        self.ctx = {}  # Backward pass variables.
        self.parameters = {}  # Anything requiring a gradient.

    def add_parameter(
        self, name, init_fn: Callable | np.ndarray, shard_axis: int | None = None
    ) -> None:
        """Add a parameter, any numpy array that also has a gradient."""
        data = init_fn() if callable(init_fn) else init_fn
        setattr(
            self,
            name,
            Parameter(data=data, gradient=np.zeros_like(data), shard_axis=shard_axis),
        )
        self.parameters[name] = getattr(self, name)

    def to_dict(self):
        """Return a dictionary representation of this layer."""
        return {name: asdict(getattr(self, name)) for name in self.parameters}

    def from_dict(self, layer_dict: dict[str, dict]):
        for name, parameter in layer_dict.items():
            self.add_parameter(name, parameter["data"], parameter["shard_axis"])

    def scatter(self):
        """Scatter this layer in place."""
        return

    def all_gather(self):
        """All-gather this layer in place."""
        return



class Linear(Layer):
    def __init__(self, d_in: int, d_out: int, shard_axis: int | None = None):
        super().__init__()
        self.add_parameter("weight", np.zeros((d_in, d_out)), shard_axis)
        self.add_parameter("bias", np.zeros((d_out,)), shard_axis)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.einsum("bsd, dh -> bsh", inputs, self.weight) + self.bias
        self.ctx["inputs"] = inputs
        return outputs

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        weight_gradient = np.einsum("bsd, bsh -> dh", self.ctx["inputs"], d_out)
        bias_gradient = d_out.sum(axis=(0, 1))
        d_out = np.einsum("bsh, dh -> bsd", d_out, self.weight)
        return d_out
