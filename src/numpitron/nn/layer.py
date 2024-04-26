from typing import Callable
import numpy as np
from dataclasses import dataclass, asdict, replace

from numpitron import distributed as npdist


@dataclass(frozen=True)
class Parameter:
    data: np.ndarray
    gradient: np.ndarray
    shard_axis: int | None = None


class Layer:
    def __init__(self):
        self.ctx = {}  # Backward pass variables.
        self.parameters = {}  # Anything requiring a gradient.
        self.is_scattered = False

    def add_parameter(
        self, name, init_fn: Callable | np.ndarray, shard_axis: int | None = None
    ) -> None:
        """Add a parameter, any numpy array that also has a gradient."""
        data = init_fn() if callable(init_fn) else init_fn

        if shard_axis is not None:
            assert shard_axis <= len(
                data.shape
            ), f"Cannot shard {shard_axis} on {len(data.shape)} dims."

        setattr(self, name, Parameter(data=data, gradient=None, shard_axis=shard_axis))
        self.parameters[name] = getattr(self, name)

    def update_parameter(self, name: str, **updates):
        current_parameter = getattr(self, name)
        setattr(self, name, replace(current_parameter, **updates))
        self.parameters[name] = getattr(self, name)

    def to_dict(self):
        """Return a dictionary representation of this layer."""
        return {name: asdict(getattr(self, name)) for name in self.parameters}

    @classmethod
    def from_dict(cls, layer_dict: dict[str, dict], **kwargs):
        layer = cls(**kwargs)
        for name, parameter in layer_dict.items():
            layer.add_parameter(name, parameter["data"], parameter["shard_axis"])
        return layer

    def scatter(self, src: int = 0):
        """Scatter this layer in place."""
        assert not self.is_scattered, "Cannot scatter an already scattered layer."
        for name, parameter in self.parameters.items():
            if parameter.shard_axis is None:
                continue

            update = {}

            new_shape = list(parameter.data.shape)
            new_shape[parameter.shard_axis] //= npdist.tensor_parallel_size()
            new_data = np.zeros(new_shape)
            npdist.scatter(
                parameter.data,
                new_data,
                parameter.shard_axis,
                src,
                npdist.tensor_parallel_group(),
            )
            update["data"] = new_data

            if parameter.gradient is not None:
                new_gradient = np.zeros(new_shape)
                npdist.scatter(
                    parameter.gradient,
                    new_gradient,
                    parameter.shard_axis,
                    src,
                    npdist.tensor_parallel_group(),
                )
                update["gradient"] = new_gradient

            self.update_parameter(name, **update)
        self.is_scattered = True

    def all_gather(self):
        """All-gather this layer in place."""
        for name, parameter in self.parameters.items():
            if parameter.shard_axis is None:
                continue

            update = {}

            new_shape = list(parameter.data.shape)
            new_shape[parameter.shard_axis] *= npdist.tensor_parallel_size()

            print(new_shape)
            new_data = np.zeros(new_shape)
            npdist.all_gather(
                parameter.data,
                new_data,
                parameter.shard_axis,
                npdist.tensor_parallel_group(),
            )
            update["data"] = new_data

            if parameter.gradient is not None:
                new_gradient = np.zeros(new_shape)
                npdist.all_gather(
                    parameter.gradient,
                    new_gradient,
                    parameter.shard_axis,
                    npdist.tensor_parallel_group(),
                )
                update["gradient"] = new_gradient

            self.update_parameter(name, **update)
        self.is_scattered = False
