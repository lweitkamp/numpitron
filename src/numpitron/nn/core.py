from abc import ABC

import numpy as np
from numpy.random import Generator


class Layer(ABC):
    def __init__(self, name: str, dtype):
        self.name = name
        self.dtype = dtype

    def init_params(self, rng: Generator) -> dict:
        return {}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        return {}, inputs

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        return {}, d_out

    def __call__(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        return self.forward(params, inputs)


class Sequential(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = []

    def init_params(self, rng: Generator) -> list[dict[str, np.ndarray] | None]:
        return {layer.name: layer.init_params(rng) for layer in self.layers}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        ctxs = {}
        for layer in self.layers:
            if len(params[layer.name]) > 0:
                print(params[layer.name].keys())
            ctx, inputs = layer(params[layer.name], inputs)
            ctxs[layer.name] = ctx
        return ctxs, inputs

    def backward(
        self, ctx: dict[str, dict], d_out: np.ndarray,
    ) -> tuple[dict[str, dict], np.ndarray]:
        gradients = {}
        for layer in self.layers[::-1]:
            gradient, d_out = layer.backward(ctx[layer.name], d_out)
            gradients[layer.name] = gradient
        return gradients, d_out
