import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from numpitron import distributed as npdist
from numpitron.nn import InputEmbedding, OutputEmbedding, PositionalEncoding

npdist.init(tp_size=npdist.world_size())


def test_input_pytorch():
    b, s, d, v = 32, 64, 128, 16

    rng = np.random.default_rng(42)

    inputs = rng.integers(0, v, (b, s))
    inputs_torch = torch.from_numpy(inputs)

    input_embedding = InputEmbedding(d, v)
    input_embedding_torch = nn.Embedding(v, d)

    input_embedding_torch.weight = Parameter(
        torch.from_numpy(input_embedding.embedding.data.T)
    )

    outputs = input_embedding(inputs)
    outputs_torch = input_embedding_torch(inputs_torch)

    outputs_torch.sum().backward()
    _ = input_embedding.backward(np.ones_like(outputs))

    np.testing.assert_allclose(
        outputs, outputs_torch.detach().numpy(), atol=1e-6, rtol=1e-2
    )
    np.testing.assert_allclose(
        input_embedding.embedding.gradient.T,
        input_embedding_torch.weight.grad,
        atol=1e-3,
        rtol=1e-2,
    )
