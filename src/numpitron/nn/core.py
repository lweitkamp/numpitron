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
    ) -> tuple[np.ndarray, dict]:
        return inputs

    def backward(self, grads: np.ndarray, ctx: dict) -> tuple[np.ndarray, dict]:
        return grads

    def __call__(self, params: dict[str, np.ndarray], inputs: np.ndarray) -> dict:
        return self.forward(params, inputs)


class Sequential(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = []

    def init_params(self, rng: Generator) -> list[dict[str, np.ndarray] | None]:
        return [layer.init_params(rng) for layer in self.layers]

    def forward(
        self, params: list[dict[str, np.ndarray] | None], inputs: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        ctxs = []
        for layer, param in zip(self.layers, params):
            inputs, ctx = layer(param, inputs)
            ctxs.append(ctx)
        return inputs, ctxs

    def backward(self, d_out: np.ndarray, ctx: list[dict]) -> tuple[np.ndarray, list[dict]]:
        gradients = []
        for layer, ctx in zip(self.layers[::-1], ctx[::-1]):
            d_out, gradient = layer.backward(d_out, ctx)
            gradients.insert(0, gradient)
        return d_out, gradients
