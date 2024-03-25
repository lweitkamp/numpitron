import numpy as np
import pytest
import torch
from numpy.random import Generator
from torch import nn, optim
from torch.nn import Parameter

from numpitron.optimizer import Adam
from numpitron.nn import Linear


TEST_SHAPES = [
    (1, 1, 8),
    (3, 1, 8),
    (1, 3, 8),
    (3, 3, 8),
]


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng(42)


@pytest.mark.parametrize("batch_size,seq_len,d_model", [shape for shape in TEST_SHAPES])
def test_adam_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    linear = Linear(input_dim=d_model, output_dim=d_model * 4)
    params = linear.init_params(rng=rng)
    linear_torch = nn.Linear(d_model, d_model * 4)

    linear_torch.weight = Parameter(torch.from_numpy(params["weight"].T))
    linear_torch.bias = Parameter(torch.from_numpy(params["bias"]))

    optimizer = Adam(1e-4, betas=(0.9, 0.999), eps=1e-8)
    state = optimizer.init_state(params)
    optimizer_torch = optim.Adam(
        linear_torch.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8
    )
    optimizer_torch.zero_grad()

    ctx, out = linear(params, inputs)
    out_torch = linear_torch(inputs_torch)

    out_torch.sum().backward()
    optimizer_torch.step()

    grads, d_out = linear.backward(ctx, np.ones_like(out))
    state, params = optimizer.step(state, grads, params)

    np.testing.assert_allclose(
        params["weight"].T, linear_torch.weight.detach().numpy(), atol=1e-4
    )
    np.testing.assert_allclose(
        params["bias"], linear_torch.bias.detach().numpy(), atol=1e-4
    )
