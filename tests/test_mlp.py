from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import Parameter

from numpitron import distributed as npdist
from numpitron.nn import MLP

npdist.init(tp_size=npdist.world_size())


def test_pytorch():
    b, s, d = 32, 64, 128

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(b * s, d)
    inputs_torch.requires_grad = True

    mlp = MLP(d, d, d, rng=rng)
    mlp_torch = nn.Sequential(
        nn.Linear(d, d * 4),
        nn.ReLU(),
        nn.Linear(d * 4, d),
    )

    mlp_torch[0].weight = Parameter(torch.from_numpy(mlp.column_linear.weight.data.T))
    mlp_torch[0].bias = Parameter(torch.from_numpy(mlp.column_linear.bias.data))
    mlp_torch[2].weight = Parameter(torch.from_numpy(mlp.row_linear.weight.data.T))
    mlp_torch[2].bias = Parameter(torch.from_numpy(mlp.row_linear.bias.data))

    outputs = mlp(inputs)
    outputs_torch = mlp_torch(inputs_torch)

    d_out = mlp.backward(np.ones_like(outputs))
    outputs_torch.sum().backward()

    np.testing.assert_allclose(
        outputs.reshape(b * s, d), outputs_torch.detach().numpy(), atol=1e-6, rtol=1e-2
    )
    np.testing.assert_allclose(
        mlp.row_linear.weight.gradient.T,
        mlp_torch[2].weight.grad,
        atol=1e-6,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        mlp.row_linear.bias.gradient, mlp_torch[2].bias.grad, atol=1e-6, rtol=1e-2
    )
    np.testing.assert_allclose(
        mlp.column_linear.weight.gradient.T,
        mlp_torch[0].weight.grad,
        atol=1e-6,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        mlp.column_linear.bias.gradient, mlp_torch[0].bias.grad, atol=1e-6, rtol=1e-2
    )
    np.testing.assert_allclose(
        d_out.reshape(inputs_torch.grad.shape),
        inputs_torch.grad.detach().numpy(),
        atol=1e-6,
        rtol=1e-2,
    )


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_tensor_parallel():
    b, s, d = 32, 64, 128

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d), dtype=np.float32)
    mlp = MLP(d, d * 4, d, rng=rng)
    mlp_tp = deepcopy(mlp)

    mlp_tp.scatter()

    out = mlp(inputs)
    out_tp = mlp_tp(inputs)

    d_out = mlp.backward(np.ones_like(out))
    d_out_tp = mlp_tp.backward(np.ones_like(out_tp))

    np.testing.assert_allclose(out, out_tp, rtol=1e-5)
    np.testing.assert_allclose(d_out, d_out_tp, rtol=1e-5)
