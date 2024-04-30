import numpy as np
import torch
import pytest
from torch import nn
from torch.nn import Parameter

from numpitron import distributed as npdist
from numpitron.nn import InputEmbedding, OutputEmbedding

npdist.init(tp_size=npdist.world_size())


def test_input_output_embedding_pytorch():
    b, s, d, v = 32, 64, 128, 16

    rng = np.random.default_rng(42)

    inputs = rng.integers(0, v, (b, s))
    inputs_torch = torch.from_numpy(inputs)

    input_embedding = InputEmbedding(d, v)
    output_embedding = OutputEmbedding(input_embedding)
    input_embedding_torch = nn.Embedding(v, d)

    input_embedding_torch.weight = Parameter(
        torch.from_numpy(input_embedding.embedding.data.T)
    )

    outputs1 = input_embedding(inputs)
    outputs2 = output_embedding(outputs1)
    outputs1_torch = input_embedding_torch(inputs_torch)
    outputs2_torch = outputs1_torch @ input_embedding_torch.weight.T

    outputs2_torch.sum().backward()
    d_out1 = output_embedding.backward(np.ones_like(outputs2))
    _ = input_embedding.backward(d_out1)

    np.testing.assert_allclose(outputs1, outputs1_torch.detach().numpy())
    np.testing.assert_allclose(outputs2, outputs2_torch.detach().numpy())
    np.testing.assert_allclose(
        input_embedding.embedding.gradient.T, input_embedding_torch.weight.grad
    )


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_tensor_parallel_input_output_embedding_pytorch():
    b, s, d, v = 32, 64, 128, 16

    rng = np.random.default_rng(42)

    inputs = rng.integers(0, v, (b, s))
    inputs_torch = torch.from_numpy(inputs)

    input_embedding = InputEmbedding(d, v)
    output_embedding = OutputEmbedding(input_embedding)
    input_embedding_torch = nn.Embedding(v, d)

    input_embedding_torch.weight = Parameter(
        torch.from_numpy(input_embedding.embedding.data.T)
    )

    input_embedding.scatter()

    outputs1 = input_embedding(inputs)
    outputs2 = output_embedding(outputs1)
    outputs1_torch = input_embedding_torch(inputs_torch)
    outputs2_torch = outputs1_torch @ input_embedding_torch.weight.T

    outputs2_torch.sum().backward()
    d_out1 = output_embedding.backward(np.ones_like(outputs2))
    _ = input_embedding.backward(d_out1)

    input_embedding.all_gather()

    np.testing.assert_allclose(outputs1, outputs1_torch.detach().numpy())
    # np.testing.assert_allclose(outputs2, outputs2_torch.detach().numpy())
    # np.testing.assert_allclose(
    #     input_embedding.embedding.gradient.T, input_embedding_torch.weight.grad
    # )
