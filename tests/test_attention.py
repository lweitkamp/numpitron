import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import Parameter

from numpitron import distributed as npdist
from numpitron.nn import Attention

npdist.init(tp_size=npdist.world_size())


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_tensor_parallel_attention():
    b, s, d, n = 16, 32, 64, 8

    rng = np.random.default_rng(42)

    inputs = (rng.random((b, s, d)).astype(np.float32) + 1) / d
    attention = Attention(
        d, n, d // n, weight_init="scaled_normal", scale=1 / (b * s * d), rng=rng
    )
    attention_tp = Attention(
        d, n, d // n, weight_init="scaled_normal", scale=1 / (b * s * d), rng=rng
    )

    attention_tp.qkv_projection.update_parameter(
        "weight", data=attention.qkv_projection.weight.data
    )
    attention_tp.out_projection.update_parameter(
        "weight", data=attention.out_projection.weight.data
    )

    attention_tp.scatter()

    out = attention(inputs)
    out_tp = attention_tp(inputs)

    d_out = attention.backward(np.ones_like(out))
    d_out_tp = attention_tp.backward(np.ones_like(out_tp))

    np.testing.assert_allclose(
        out,
        out_tp,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        d_out,
        d_out_tp,
        atol=1e-6,
        rtol=1e-6,
    )


def test_pytorch():
    b, s, d, n = 2, 16, 32, 4

    rng = np.random.default_rng(42)

    inputs = (rng.random((b, s, d)).astype(np.float32) + 1) / d
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    attention = Attention(d, n, d // n, weight_init="init_for_testing", rng=rng)
    attention_torch = nn.MultiheadAttention(d, n, bias=False, batch_first=True)
    attention_torch.in_proj_weight = Parameter(
        torch.from_numpy(attention.qkv_projection.weight.data).T
    )
    attention_torch.out_proj.weight = Parameter(
        torch.from_numpy(attention.out_projection.weight.data)
    )

    out = attention(inputs)
    out_torch, attn_output_weights = attention_torch(
        inputs_torch,
        inputs_torch,
        inputs_torch,
        is_causal=True,
        attn_mask=torch.from_numpy(~attention.ctx["mask"].squeeze((0, 1))),
        average_attn_weights=False,
    )

    np.testing.assert_allclose(
        attn_output_weights.detach().numpy(),
        attention.ctx["attention_weights"],
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        out,
        out_torch.detach().numpy(),
        atol=1e-2,
        rtol=1e-2,
    )

    out_torch.sum().backward()
    attention.backward(np.ones_like(out))

    np.testing.assert_allclose(
        attention.out_projection.weight.gradient.T,
        attention_torch.out_proj.weight.grad,
        atol=0.1,
        rtol=0.1,
    )
    np.testing.assert_allclose(
        attention.qkv_projection.weight.gradient.sum(),
        attention_torch.in_proj_weight.grad.sum(),
        atol=1e-3,
        rtol=1e-3,
    )
