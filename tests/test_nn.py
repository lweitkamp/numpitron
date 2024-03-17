"""Unit tests for the neural network library. To test it out, ensure pytorch
is installed."""


import numpy as np
import pytest
import torch
from numpy.random import Generator
from torch import nn
from torch.nn import Parameter

from numpitron.nn import MLP, InputEmbedding, Linear, OutputEmbedding


TEST_SHAPES = [
    (1, 1, 8),
    (3, 1, 8),
    (1, 3, 8),
    (3, 3, 8),
]


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng(42)


@pytest.mark.parametrize("batch_size,seq_len,d_model", TEST_SHAPES)
def test_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    linear = Linear(input_dim=d_model, output_dim=d_model * 4)
    params = linear.init_params(rng=rng)
    linear_torch = nn.Linear(d_model, d_model * 4)

    # Transfer weights.
    linear_torch.weight = Parameter(torch.from_numpy(params["weight"].T))
    linear_torch.bias = Parameter(torch.from_numpy(params["bias"]))

    # Forward through both models.
    out, ctx = linear(params, inputs)
    out_torch = linear_torch(inputs_torch)

    # Backward through both models.
    out_torch.sum().backward()
    _, gradients = linear.backward(np.ones_like(out), ctx)

    # Forward pass should be (approx) equal.
    np.testing.assert_allclose(
        out.reshape(batch_size * seq_len, d_model * 4),
        out_torch.detach().numpy(),
        atol=1e-5,
    )

    # Gradients calculated should be (approx) equal.
    np.testing.assert_allclose(
        gradients["weight"].T,
        linear_torch.weight.grad,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        gradients["bias"],
        linear_torch.bias.grad,
        atol=1e-5,
    )


@pytest.mark.parametrize("batch_size,seq_len,d_model", TEST_SHAPES)
def test_mlp(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    mlp = MLP(d_model, d_model * 4)
    params = mlp.init_params(rng=rng)
    mlp_torch = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.ReLU(),
        nn.Linear(d_model * 4, d_model),
    )

    # Transfer weights.
    mlp_torch[0].weight = Parameter(torch.from_numpy(params["Linear1"]["weight"].T))
    mlp_torch[0].bias = Parameter(torch.from_numpy(params["Linear1"]["bias"]))
    mlp_torch[2].weight = Parameter(torch.from_numpy(params["Linear2"]["weight"].T))
    mlp_torch[2].bias = Parameter(torch.from_numpy(params["Linear2"]["bias"]))

    # Forward through both models.
    out, ctx = mlp(params, inputs)
    out_torch = mlp_torch(inputs_torch)

    # Backward through both models.
    out_torch.sum().backward()
    _, gradients = mlp.backward(np.ones_like(out), ctx)

    # # Forward pass should be (approx) equal.
    np.testing.assert_allclose(
        out.reshape(batch_size * seq_len, d_model),
        out_torch.detach().numpy(),
        atol=1e-5,
    )

    # Gradients calculated should be (approx) equal.
    np.testing.assert_allclose(
        gradients[2]["weight"].T,
        mlp_torch[2].weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        gradients[2]["bias"],
        mlp_torch[2].bias.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        gradients[0]["weight"].T,
        mlp_torch[0].weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        gradients[0]["bias"],
        mlp_torch[0].bias.grad,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,vocab_size", [shape + (20,) for shape in TEST_SHAPES]
)
def test_input_embedding(
    batch_size: int,
    seq_len: int,
    d_model: int,
    vocab_size: int,
    rng: Generator,
):
    inputs = rng.integers((batch_size, seq_len, vocab_size))
    inputs_torch = torch.from_numpy(inputs)

    embedding = InputEmbedding(d_model, vocab_size)
    params = embedding.init_params(rng=rng)
    embedding_torch = nn.Embedding(vocab_size, d_model)

    # Transfer weights.
    embedding_torch.weight = Parameter(torch.from_numpy(params["embedding"].T))

    # Forward through both models.
    out, ctx = embedding(params, inputs)
    out_torch = embedding_torch(inputs_torch)

    # Backward through both models.
    _, gradients = embedding.backward(np.ones_like(out), ctx)
    out_torch.sum().backward()

    # Forward pass should be (approx) equal.
    np.testing.assert_allclose(
        out,
        out_torch.detach().numpy(),
        atol=1e-5,
    )
    # Gradients calculated should be (approx) equal.
    np.testing.assert_allclose(
        gradients["embedding"].T,
        embedding_torch.weight.grad,
        atol=1e-5,
    )
