from copy import deepcopy
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

    inputs = np.ones((b, s, d))
    linear = Linear(d, d, weight_init="ones", bias_init="ones")
    outputs = linear(inputs)

    np.testing.assert_allclose(outputs.sum(), b * s * d * d + b * s * d)


def test_linear_no_bias():
    b, s, d = 32, 64, 128

    inputs = np.ones((b, s, d))
    linear = Linear(d, d, use_bias=False, weight_init="ones")
    outputs = linear(inputs)

    np.testing.assert_allclose(outputs.sum(), b * s * d * d)


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_row_linear():
    b, s, d = 32, 64, 128
    size, rank = npdist.world_size(), npdist.tensor_parallel_rank()
    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d)).astype(np.float32)
    row_linear = Linear(d, d, weight_shard_axis=0, rng=rng, bias_init="ones")
    linear = deepcopy(row_linear)

    inputs_scattered = np.empty((b, s, d // size), dtype=np.float32)
    npdist.scatter(
        inputs, inputs_scattered, axis=-1, group=npdist.tensor_parallel_group()
    )

    row_linear.use_bias = rank == 0
    row_linear.scatter()

    outputs_tp = row_linear(inputs_scattered)
    outputs = linear(inputs)

    npdist.all_reduce(outputs_tp, group=npdist.tensor_parallel_group())

    np.testing.assert_allclose(outputs, outputs_tp, atol=1e-6)


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_column_linear():
    b, s, d = 32, 64, 128
    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d)).astype(np.float32)
    column_linear = Linear(
        d, d, weight_shard_axis=1, bias_shard_axis=0, rng=rng, bias_init="ones"
    )
    linear = deepcopy(column_linear)

    column_linear.scatter()

    outputs_tp = column_linear(inputs)
    outputs = linear(inputs)

    outputs_tp_gathered = np.empty_like(outputs, dtype=np.float32)
    npdist.all_gather(
        outputs_tp, outputs_tp_gathered, axis=-1, group=npdist.tensor_parallel_group()
    )

    np.testing.assert_allclose(outputs, outputs_tp_gathered)


def test_pytorch():
    b, s, d = 32, 64, 128

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(b * s, d)
    inputs_torch.requires_grad = True

    linear = Linear(d, d, weight_init="init_for_testing", bias_init="ones", rng=rng)
    linear_torch = nn.Linear(d, d)

    linear_torch.weight = Parameter(torch.from_numpy(linear.weight.data.T))
    linear_torch.bias = Parameter(torch.from_numpy(linear.bias.data))

    outputs = linear(inputs)
    outputs_torch = linear_torch(inputs_torch)

    d_out = linear.backward(np.ones_like(outputs))
    outputs_torch.sum().backward()

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
