from abc import abstractmethod
from dataclasses import asdict, dataclass, replace

import numpy as np
from numpy.random import Generator

from numpitron import distributed as npdist


def weight_init_fn(
    name: str = "default",
    rng: Generator | None = None,
    scale: float = 1.0,
    **kwargs,
):
    def zeros(shape: tuple[int, ...]):
        return np.zeros(shape).astype(np.float32)

    def ones(shape: tuple[int, ...]):
        return np.ones(shape).astype(np.float32)

    def in_projection(shape: tuple[int, int]):
        assert rng is not None
        in_, out_ = shape
        weight = rng.random((in_, out_)) * 3 / (in_ * out_)
        return weight.astype(np.float32)

    def scaled_normal(shape: tuple[int, ...]):
        assert rng is not None
        weight = rng.random(shape) * scale
        return weight.astype(np.float32)

    def init_for_testing(shape: tuple[int, ...]):
        assert rng is not None
        weight = (rng.random(shape) + 1) / max(shape)
        return weight.astype(np.float32)

    options = {
        "zeros": zeros,
        "ones": ones,
        "in_projection": in_projection,
        "scaled_normal": scaled_normal,
        "init_for_testing": init_for_testing,
    }
    return options[name]


@dataclass(frozen=True)
class Parameter:
    data: np.ndarray
    gradient: np.ndarray
    shard_axis: int | None = None


class Layer:
    def __init__(self, **kwargs):
        self.ctx = {}  # Backward pass variables.
        self.parameters = {}  # Anything requiring a gradient.
        self.settings = {}  # Anything that is required for initializing itself.
        self.is_scattered = False

    @abstractmethod
    def forward(self):
        ...

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def add_parameter(
        self, name, data: np.ndarray, shard_axis: int | None = None
    ) -> None:
        """Add a parameter, any numpy array that also has a gradient."""
        if shard_axis is not None:
            assert shard_axis <= len(
                data.shape
            ), f"Cannot shard {shard_axis} on {len(data.shape)} dims."
        setattr(self, name, Parameter(data=data, gradient=None, shard_axis=shard_axis))
        self.parameters[name] = getattr(self, name)

    def add_setting(self, name, value):
        setattr(self, name, value)
        self.settings[name] = value

    def add_settings(self, settings: dict):
        for key, value in settings.items():
            self.add_setting(key, value)

    def update_parameter(self, name: str, **updates):
        current_parameter = getattr(self, name)
        setattr(self, name, replace(current_parameter, **updates))
        self.parameters[name] = getattr(self, name)

    def to_dict(self):
        """Return a dictionary representation of this layer."""
        parameters = {name: asdict(getattr(self, name)) for name in self.parameters}
        return {"settings": self.settings, "parameters": parameters}

    @classmethod
    def from_dict(cls, layer_dict: dict[str, dict]):
        settings, parameters = layer_dict["settings"], layer_dict["parameters"]
        settings |= {"weight_init": "zeros", "bias_init": "zeros"}
        layer = cls(**settings)

        for name, parameter in parameters.items():
            layer.add_parameter(name, parameter["data"], parameter["shard_axis"])
        return layer

    def scatter(self, src: int = 0):
        """Scatter this layer in place."""
        assert not self.is_scattered, "Cannot scatter an already scattered layer."
        for name, parameter in self.parameters.items():
            if parameter.shard_axis is None:
                continue

            update = {}

            # New shape depends on the division of the tensor parallel size.
            new_shape = np.array_split(
                parameter.data, npdist.tensor_parallel_size(), axis=parameter.shard_axis
            )[npdist.tensor_parallel_rank()].shape
            new_data = np.zeros(new_shape, dtype=parameter.data.dtype)

            npdist.scatter(
                parameter.data,
                new_data,
                parameter.shard_axis,
                src,
                npdist.tensor_parallel_group(),
            )
            update["data"] = new_data

            if parameter.gradient is not None:
                new_gradient = np.zeros(new_shape, dtype=parameter.gradient.dtype)
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

            new_data = np.zeros(new_shape, dtype=parameter.data.dtype)
            npdist.all_gather(
                parameter.data,
                new_data,
                parameter.shard_axis,
                npdist.tensor_parallel_group(),
            )
            update["data"] = new_data

            if parameter.gradient is not None:
                new_gradient = np.zeros(new_shape, dtype=parameter.gradient.dtype)
                npdist.all_gather(
                    parameter.gradient,
                    new_gradient,
                    parameter.shard_axis,
                    npdist.tensor_parallel_group(),
                )
                update["gradient"] = new_gradient

            self.update_parameter(name, **update)
        self.is_scattered = False
