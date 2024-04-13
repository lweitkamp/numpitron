# flake8: noqa
from numpitron.nn.activation import ReLU, Softmax, softmax
from numpitron.nn.attention import Attention
from numpitron.nn.loss import SoftmaxCrossEntropy
from numpitron.nn.embedding import InputEmbedding, OutputEmbedding, PositionalEmbedding
from numpitron.nn.linear import Linear
from numpitron.nn.mlp import MLP
from numpitron.nn.normalization import LayerNorm

from numpitron.tensor_parallel.loss import TensorParallelSoftmaxCrossEntropy


def from_config(cfg: dict):
    """Load a Layer from a config file."""
    layer = {
        "SoftmaxCrossEntropy": SoftmaxCrossEntropy,
        # can do better 
        "TensorParallelSoftmaxCrossEntropy": TensorParallelSoftmaxCrossEntropy,
    }[cfg["layer_type"]]
    return layer(**cfg["layer_config"])
