from typing import Self
import numpy as np

from numpitron.nn.model import Model
from numpitron.nn.core import Layer, weight_init_fn


class AdamParams(Layer):
    def __init__(self, shape: tuple[int, ...], shard_axis: int | None = None, **kwargs):
        super().__init__()
        self.add_parameter("momentum", weight_init_fn("zeros")(shape), shard_axis)
        self.add_parameter("velocity", weight_init_fn("zeros")(shape), shard_axis)


def adam_update(
    parameter: str,
    layer: Layer,
    adam_state: Layer,
    learning_rate: float,
    beta1: float,
    beta2: float,
    timestep: int,
    eps: float,
) -> None:
    layer_parameter = layer.parameters[parameter]

    adam_state.update_parameter(
        "momentum",
        data=beta1 * adam_state.momentum.data + (1 - beta1) * layer_parameter.gradient,
    )

    adam_state.update_parameter(
        "velocity",
        data=beta2 * adam_state.velocity.data
        + (1 - beta2) * np.power(layer_parameter.gradient, 2),
    )

    momentum = adam_state.momentum.data / (1 - beta1**timestep)
    velocity = adam_state.velocity.data / (1 - beta2**timestep)
    update = (learning_rate * momentum) / (np.sqrt(velocity) + eps)
    layer.update_parameter(parameter, data=layer_parameter.data - update, gradient=None)


class Adam:
    def __init__(
        self,
        model: Model,
        learning_rate: float,
        beta1: float = 0.99,
        beta2: float = 0.999,
        eps: float = 1e-7,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.timestep = 1
        self.init()

    def init(self):
        # Retrieve references of each layer.
        self.layers = self.model.get_layers()

        def _init_params(layer: dict) -> dict:
            params = {}
            for n, l in layer.items():
                if isinstance(l, Layer):
                    params[n] = {
                        param_name: AdamParams(param.data.shape, param.shard_axis)
                        for param_name, param in l.parameters.items()
                    }
                elif isinstance(l, dict):
                    params[n] = _init_params(l)
            return params

        self.parameters = _init_params(self.layers)

    def step(self, layer, params):
        if layer is None and params is None:
            layer, params = self.layers, self.parameters

        if isinstance(layer, Layer):
            for name, optim_parameters in params.items():
                adam_update(
                    name,
                    layer,
                    optim_parameters,
                    self.learning_rate,
                    self.beta1,
                    self.beta2,
                    self.timestep,
                    self.eps,
                )
            return
        for key in params:
            self.step(layer[key], params[key])

    def to_dict(self):
        def _to_dict(params) -> dict:
            out = {}
            for key, value in params.items():
                if isinstance(value, dict):
                    out[key] = _to_dict(value)
                elif isinstance(value, Layer):
                    out[key] = value.to_dict()
                else:
                    print(key, value)
            return out

        state = {
            "parameters": _to_dict(self.parameters),
            "timestep": self.timestep,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }
        return state

    @classmethod
    def from_dict(cls, state: dict, model: Model) -> Self:
        parameters = state.pop("parameters")
        timestep = state.pop("timestep")
        optimizer = cls(model, **state)

        # Fix parameters.
        def recursive(something):
            out = {}
            for key, value in something.items():
                if "settings" in value and "parameters" in value:
                    momentum = value["parameters"]["momentum"]
                    velocity = value["parameters"]["velocity"]
                    out[key] = AdamParams(
                        momentum["data"].shape, momentum["shard_axis"]
                    )
                    out[key].update_parameter("momentum", data=momentum["data"])
                    out[key].update_parameter("velocity", data=velocity["data"])
                else:
                    out[key] = recursive(value)
            return out

        optimizer.parameters = recursive(parameters)
        optimizer.timestep = timestep
        return optimizer
