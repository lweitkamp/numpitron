from typing import Self
import numpy as np

from numpitron.nn import Transformer
from numpitron.nn.model import Model
from numpitron.nn.core import Parameter, Layer


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
                        param_name: {
                            "velocity": np.zeros_like(param.data),
                            "momentum": np.zeros_like(param.data),
                        }
                        for param_name, param in l.parameters.items()
                    }
                elif isinstance(l, dict):
                    params[n] = _init_params(l)
            return params

        self.parameters = _init_params(self.layers)

    def step(self):
        def _step(layer, params):
            if isinstance(layer, Layer):
                for name, optim_parameters in params.items():
                    layer_parameter = layer.parameters[name]

                    np.copyto(
                        optim_parameters["momentum"],
                        self.beta1 * optim_parameters["momentum"]
                        + (1 - self.beta1) * layer_parameter.gradient,
                    )
                    np.copyto(
                        optim_parameters["velocity"],
                        self.beta2 * optim_parameters["velocity"]
                        + (1 - self.beta2) * np.power(layer_parameter.gradient, 2),
                    )

                    momentum = optim_parameters["momentum"] / (
                        1 - self.beta1**self.timestep
                    )
                    velocity = optim_parameters["velocity"] / (
                        1 - self.beta2**self.timestep
                    )
                    update = (self.learning_rate * momentum) / (
                        np.sqrt(velocity) + self.eps
                    )
                    layer.update_parameter(name, data=layer_parameter.data - update, gradient=None)
                    return

            for key in params:
                _step(layer[key], params[key])

        _step(self.layers, self.parameters)
        
    def to_dict(self):
        state = {
            "parameters": self.parameters,
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
        optimizer.parameters = parameters
        optimizer.timestep = timestep
        return optimizer
