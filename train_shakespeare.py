import numpy as np

from numpitron.optimizer import Adam
from numpitron.nn import Transformer, softmax_cross_entropy


def train_step(
    transformer: Transformer, optimizer: Adam, x: np.ndarray, y: np.ndarray
) -> float:
    y_hat = transformer(x)
    loss, d_out = softmax_cross_entropy(y_hat, y)
    transformer.backward(d_out)
    optimizer.step()
    return loss.mean()


def validation_step(transformer: Transformer, x: np.ndarray, y: np.ndarray) -> float:
    y_hat = transformer(x)
    loss, _ = softmax_cross_entropy(y_hat, y)
    return loss.mean()


