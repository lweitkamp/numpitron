import numpy as np


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

    def init_state(self, parameters: dict) -> dict:
        def _add_hparams(p):
            return {
                key: {
                    "momentum": np.zeros_like(value),
                    "velocity": np.zeros_like(value),
                }
                if isinstance(value, np.ndarray)
                else _add_hparams(value)
                for key, value in p.items()
            }

        state = {
            "learning_rate": self.learning_rate,
            "timestep": 0,
            "beta0": self.betas[0],
            "beta1": self.betas[1],
            "eps": self.eps,
            "state": _add_hparams(parameters),
        }
        return state

    def step(
        self, optimizer_state: dict, gradients: dict, parameters: dict
    ) -> tuple[dict, dict]:
        lr = optimizer_state["learning_rate"]
        b0 = optimizer_state["beta0"]
        b1 = optimizer_state["beta1"]
        timestep = optimizer_state["timestep"] + 1
        eps = optimizer_state["eps"]

        def update(o, g, p):
            new_o, new_p = {}, {}
            for key, value in p.items():
                if isinstance(p[value], np.ndarray):
                    new_o["momentum"] = b0 * o["momentum"] + (1 - b0) * g["gradient"]
                    new_o["velocity"] = b1 * o["velocity"] + (1 - b1) * np.power(
                        g["gradient"], 2
                    )

                    m = new_o["momentum"] / (1 - b0**timestep)
                    v = new_o["velocity"] / (1 - b1**timestep)
                    new_p[key] = p[key] - (lr * m / (np.sqr(v) + eps))
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
            "beta0": b0,
            "beta1": b1,
            "eps": eps,
            "state": new_optimizer_state,
        }

        return new_optimizer_state, new_parameters

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
