import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from numpitron import distributed as npdist
from numpitron.nn import LayerNorm

npdist.init(tp_size=npdist.world_size())



def test_layernorm_pytorch():
    b, s, d = 32, 64, 128

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(b * s, d)
    inputs_torch.requires_grad = True

    layer_norm = LayerNorm(d, eps=1e-5)
    layer_norm_torch = nn.LayerNorm(d, eps=1e-5)

    layer_norm_torch.weight = Parameter(torch.from_numpy(layer_norm.weight.data))
    layer_norm_torch.bias = Parameter(torch.from_numpy(layer_norm.bias.data))

    outputs = layer_norm(inputs)
    outputs_torch = layer_norm_torch(inputs_torch)

    outputs_torch.sum().backward()
    d_out = layer_norm.backward(np.ones_like(outputs))

    np.testing.assert_allclose(
        outputs, outputs_torch.detach().numpy().reshape(b, s, d), atol=1e-6, rtol=1e-2
    )
    np.testing.assert_allclose(
        layer_norm.weight.gradient, layer_norm_torch.weight.grad, atol=1e-3, rtol=1e-2
    )
    np.testing.assert_allclose(
        layer_norm.bias.gradient,
        layer_norm_torch.bias.grad,
    )
    np.testing.assert_allclose(
        d_out.reshape(inputs_torch.grad.shape),
        inputs_torch.grad.detach().numpy(),
        atol=1e-4,
        rtol=1e-2,
    )
