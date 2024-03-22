from pathlib import Path
import numpy as np
from typing import Callable
from dataclasses import dataclass


def tree_map(fn: Callable, *args) -> dict:    
    def tree_map_single(fn: Callable, inputs):
        if not isinstance(inputs, dict):
            return fn(inputs)
        return {key: tree_map_single(fn, inputs) for key in inputs}
    
    def tree_map_multiple(fn: Callable, *args):
        if not isinstance(args[0], dict):
            return fn(*args)
        return {key: tree_map_multiple(fn, *args) for key in args[0]}
    
    if len(args) == 1:
        return tree_map_single(fn, args[0])

    return tree_map_multiple(fn, *args)


@dataclass
class AdamStats:
    velocity: np.ndarray
    momentum: np.ndarray


class Adam:
    def __init__(
        self,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        
    
    @staticmethod
    def init(parameter: np.ndarray) -> dict[str, np.ndarray]:
        return AdamStats(np.zeros_like(parameter), np.zeros_like(parameter))

    def init_state(self, parameters: dict) -> dict:
        return {
            "timestep": 0,
            "beta0": self.betas[0],
            "beta1": self.betas[1],
            "eps": self.eps,
            "state": tree_map(self.init, parameters),
        }
    
    @staticmethod
    def update(optimizer_state: dict, gradients: dict, beta0: float, beta1: float, timestep: int):
        b1, b2 = self.betas

        state.momentum = b1 * state.momentum + (1 - b1) * gradients
        state.velocity = b2 * state.velocity + (1 - b2) * np.power(gradients, 2)

        momentum = state.momentum / (1 - b1**self.timestep)
        velocity = state.velocity / (1 - b2**self.timestep)
        update = self.learning_rate * momentum / (np.sqrt(velocity) + self.eps)
        state.parameter.data = state.parameter.data - update


    def step(self, optimizer_state: dict, gradients: dict, parameters: dict) -> dict:
        optimizer_state["timestep"] += 1
        
        # We need to take the optimizr state, the gradients, and the model weights.
        
        new_gradients = tree_map(self.update, gradients, optimizer_state["state"], )
        return optimizer_state, new_gradients

    # def save(self, path: Path):
    #     save_state = {}

    #     save_state["timestep"] = self.timestep
    #     save_state["learning_rate"] = self.learning_rate
    #     save_state["eps"] = self.eps
    #     save_state["beta0"], save_state["beta1"] = self.betas


    # def load(self, path: Path):
    #     save_state = np.load(path, allow_pickle=True)[()]

    #     self.timestep = save_state["timestep"]
    #     self.learning_rate = save_state["learning_rate"]
    #     self.eps = save_state["eps"]
    #     self.betas = save_state["beta0"], save_state["beta1"]
