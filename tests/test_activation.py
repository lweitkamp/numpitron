import numpy as np
import torch
from torch import nn

from numpitron import distributed as npdist
from numpitron.nn import Softmax, ReLU


npdist.init(tp_size=npdist.world_size())


def test_softmax_pytorch():
    b, s, n = 32, 64, 8

    rng = np.random.default_rng(42)

    inputs = rng.random((b, n, s, s))
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    softmax = Softmax(axis=-1)
    softmax_torch = nn.Softmax(dim=-1)

    outputs = softmax(inputs)
    outputs_torch = softmax_torch(inputs_torch)

    outputs_torch.sum().backward()
    d_out = softmax.backward(np.ones_like(outputs))

    np.testing.assert_allclose(outputs, outputs_torch.detach().numpy())
    np.testing.assert_allclose(d_out, inputs_torch.grad.detach().numpy(), atol=1e-4)


def test_relu_pytorch():
    b, s, d = 32, 64, 128

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d))
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    relu = ReLU()
    relu_torch = nn.ReLU()

    outputs = relu(inputs)
    outputs_torch = relu_torch(inputs_torch)

    outputs_torch.sum().backward()
    d_out = relu.backward(np.ones_like(outputs))

    np.testing.assert_allclose(outputs, outputs_torch.detach().numpy())
    np.testing.assert_allclose(d_out, inputs_torch.grad.detach().numpy())
