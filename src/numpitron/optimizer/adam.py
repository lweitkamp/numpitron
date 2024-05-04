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
        self.timestep = 0
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
                        self.beta1 * optim_parameters["momentum"]
                        + (1 - self.beta1) * layer_parameter.gradient,
                        optim_parameters["momentum"],
                    )
                    np.copyto(
                        self.beta2 * optim_parameters["velocity"]
                        + (1 - self.beta2) * np.power(layer_parameter.gradient, 2),
                        optim_parameters["velocity"],
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
                    layer.update_parameter(name, data=layer_parameter.data - update)
                    return

            for key in params:
                _step(layer[key], params[key])

        _step(self.layers, self.parameters)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    model = Transformer(32, 6, 3, 16, 8, rng=rng)
    adam = Adam(model, 1e-4)

    inputs = rng.integers(0, 16, (7, 8))
    outputs = model(inputs)
    model.backward(np.ones_like(outputs))
    print(adam.step())
