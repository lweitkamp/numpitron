import numpy as np


class Adam:
    """Adam optimizer, keeps track of momentum and variance of gradients."""

    def __init__(
        self,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps

    def init_state(self, parameters: dict) -> dict:
        def _add_hparams(p):
            return {
                key: {"m": np.zeros_like(value), "v": np.zeros_like(value)}
                if isinstance(value, np.ndarray)
                else _add_hparams(value)
                for key, value in p.items()
            }

        state = {
            "learning_rate": self.learning_rate,
            "timestep": 0,
            "betas": self.betas,
            "eps": self.eps,
            "state": _add_hparams(parameters),
        }
        return state

    def step(
        self, optimizer_state: dict, gradients: dict, parameters: dict
    ) -> tuple[dict, dict]:
        lr = optimizer_state["learning_rate"]
        b1, b2 = optimizer_state["betas"]
        timestep = optimizer_state["timestep"] + 1
        eps = optimizer_state["eps"]

        def update(o, g, p):
            new_o, new_p = {}, {}
            for key in p.keys():
                if isinstance(p[key], np.ndarray):
                    new_o[key] = {
                        "m": b1 * o[key]["m"] + (1 - b1) * g[key],
                        "v": b2 * o[key]["v"] + (1 - b2) * np.power(g[key], 2),
                    }

                    m = new_o[key]["m"] / (1 - b1**timestep)
                    v = new_o[key]["v"] / (1 - b2**timestep)
                    new_p[key] = p[key] - (lr * m / (np.sqrt(v) + eps))
                else:
                    _o, _p = update(o[key], g[key], p[key])
                    new_o[key] = _o
                    new_p[key] = _p
            return new_o, new_p

        new_optimizer_state, new_parameters = update(
            optimizer_state["state"], gradients, parameters
        )

        new_optimizer_state = {
            "learning_rate": lr,
            "timestep": timestep,
            "betas": optimizer_state["betas"],
            "eps": eps,
            "state": new_optimizer_state,
        }

        return new_optimizer_state, new_parameters
