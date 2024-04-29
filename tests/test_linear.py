import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import Parameter

from numpitron import distributed as npdist
from numpitron.nn import Linear

npdist.init(tp_size=npdist.world_size())


def test_linear():
    b, s, d = 32, 64, 128

    inputs = np.ones((b, d))
    linear = Linear(d, d, init_fn=lambda shape: np.ones(shape))
    outputs = linear(inputs)

    np.testing.assert_allclose(outputs.sum(), b * s * d * d + b * s * d)


def test_linear_no_bias():
    b, s, d = 32, 64, 128

    inputs = np.ones((b, s, d))
    linear = Linear(d, d, use_bias=False, init_fn=lambda shape: np.ones(shape))
    outputs = linear(inputs)

    np.testing.assert_allclose(outputs.sum(), b * s * d * d)


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_row_linear():
    b, s, d = 32, 64, 128
    size, rank = npdist.world_size(), npdist.tensor_parallel_rank()

    inputs_scattered = np.ones((b, s, d // size))
    row_linear = Linear(d, d, init_fn=lambda shape: np.ones(shape), weight_shard_axis=0)
    row_linear.use_bias = rank == 0
    row_linear.scatter()

    outputs = row_linear(inputs_scattered)

    np.testing.assert_allclose(
        outputs.sum(), b * s * (d // size) * d + (rank == 0) * b * s * d
    )
    assert outputs.shape == (b, s, d)


def test_column_linear():
    b, s, d = 32, 64, 128
    size = npdist.world_size()

    inputs = np.ones((b, s, d))
    row_linear = Linear(
        d,
        d,
        init_fn=lambda shape: np.ones(shape),
        weight_shard_axis=1,
        bias_shard_axis=0,
    )
    row_linear.scatter()

    outputs = row_linear(inputs)

    np.testing.assert_allclose(
        outputs.sum(), b * s * d * (d // size) + b * s * (d // size)
    )
    assert outputs.shape == (b, s, d // size)


def test_pytorch():
    b, s, d = 32, 64, 128

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(b * s, d)
    inputs_torch.requires_grad = True

    linear = Linear(
        d, d, init_fn=lambda shape: (rng.normal(size=shape) + 1) / shape[-1]
    )
    linear_torch = nn.Linear(d, d)

    linear_torch.weight = Parameter(torch.from_numpy(linear.weight.data.T))
    linear_torch.bias = Parameter(torch.from_numpy(linear.bias.data))

    outputs = linear(inputs)
    outputs_torch = linear_torch(inputs_torch)

    outputs_torch.sum().backward()
    d_out = linear.backward(np.ones_like(outputs))

    np.testing.assert_allclose(
        outputs, outputs_torch.detach().numpy().reshape(b, s, d), atol=1e-6, rtol=1e-2
    )
    np.testing.assert_allclose(
        linear.weight.gradient.T, linear_torch.weight.grad, atol=1e-3, rtol=1e-2
    )
    np.testing.assert_allclose(
        linear.bias.gradient,
        linear_torch.bias.grad,
    )
    np.testing.assert_allclose(
        d_out.reshape(inputs_torch.grad.shape),
        inputs_torch.grad.detach().numpy(),
        atol=1e-6,
        rtol=1e-2,
    )
