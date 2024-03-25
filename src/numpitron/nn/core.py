"""Core layer abstractions."""
from abc import ABC
from pathlib import Path

import numpy as np
from numpy.random import Generator


class Layer(ABC):
    """Layer defines that a layer should have a forward path annd a backward
    path, alongside a way to init a layer with no parameters by default.
    
    Args:
        name (str): name of the layer, crucial for sequential models.
        dtype (np.dtype): data type the layer works on.
    
    """

    def __init__(self, name: str, dtype):
        self.name = name
        self.dtype = dtype

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        """Initialize this layer's weights. Default has no weights.
        
        A weight is a dictionary of possible sub-dictionaries that
        lead to numpy arrays, denoted here as a tree of parameters typically.
        
        Args:
            rng (Generator): NumPy random number generator.
        """
        return {}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        return {}, inputs

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[dict, np.ndarray]:
        return {}, d_out

    def __call__(self, *args, **kwargs) -> tuple[dict, np.ndarray]:
        return self.forward(*args, **kwargs)


class Sequential(Layer):
    """A sequential layer makes it easier to define a list of sequential
    operations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = []

    def init_params(self, rng: Generator) -> dict[str, np.ndarray]:
        return {layer.name: layer.init_params(rng) for layer in self.layers}

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        ctxs = {}
        for layer in self.layers:
            ctx, inputs = layer(params[layer.name], inputs)
            ctxs[layer.name] = ctx
        return ctxs, inputs

    def backward(
        self,
        ctx: dict[str, dict],
        d_out: np.ndarray,
    ) -> tuple[dict[str, dict], np.ndarray]:
        gradients = {}
        for layer in self.layers[::-1]:
            gradient, d_out = layer.backward(ctx[layer.name], d_out)
            gradients[layer.name] = gradient
        return gradients, d_out


def save_params(save_path: Path | str, params: dict) -> None:
    """Store a tree of parameters in pickle form.
    
    Args:
        save_path (Path or str): Path to save to.
        params (dict): tree of parameters to save.
    """
    np.save(save_path, params, allow_pickle=True)


def load_params(load_path: Path | str) -> dict:
    """Load a tree of parameters in pickle form.
    
    Args:
        load_path (Path or str): Path to load pickle from.
    
    Returns:
        tree of parameters.
    """
    return np.load(load_path, allow_pickle=True)[()]
