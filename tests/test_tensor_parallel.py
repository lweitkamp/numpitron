"""Every tests follows the following logic:

1. Initiate parallel and non-parallel layer.
2. 'Scatter' non-parallel weights to parallel layer.
3. Run a forward + backward pass with both.
4. Check outputs + the 'Gather' results.
"""

import numpy as np
import pytest
from numpy.random import Generator

import numpitron.distributed as npdist
from numpitron import nn
from numpitron.tensor_parallel import TensorParallelAttention

world_comm = npdist.WorldCommunicator()
RANK = world_comm.world_rank
WORLD_SIZE = world_comm.world_size

comm = npdist.ParallelCommunicator(tp_size=WORLD_SIZE)


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
def test_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    n_heads: int = d_model // 2

    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)

    attention = nn.Attention(d_model, n_heads, d_model // n_heads)
    attention_tp = TensorParallelAttention(
        d_model, n_heads, d_model // n_heads, communicator=comm
    )
    params = attention.init_params(rng=rng)
    params_tp = attention_tp.init_params(rng=rng)

    for weight in ("q_projection", "k_projection", "v_projection"):
        npdist.scatter(
            params[weight]["weight"],
            params_tp[weight]["weight"],
            axis=1,
            comm=comm.tp_comm,
        )
        npdist.scatter(
            params[weight]["bias"],
            params_tp[weight]["bias"],
            axis=0,
            comm=comm.tp_comm,
        )

    npdist.scatter(
        params["out_projection"]["weight"],
        params_tp["out_projection"]["weight"],
        axis=0,
        comm=comm.tp_comm,
    )
    npdist.scatter(
        params["out_projection"]["bias"],
        params_tp["out_projection"]["bias"],
        axis=0,
        comm=comm.tp_comm,
    )

    ctx, out = attention(params, inputs)
    ctx_tp, out_tp = attention_tp(params_tp, inputs)

    grads, d_out = attention.backward(ctx, np.ones_like(out))
    grads_tp, d_out_tp = attention_tp.backward(ctx_tp, np.ones_like(out_tp))

    np.testing.assert_allclose(out, out_tp, atol=1e-5)
    # np.testing.assert_allclose(d_out, d_out_tp, rtol=1e-5, atol=1e-5)

    # for weight in ("q_projection", "k_projection", "v_projection"):
    #     x = np.zeros_like(grads[weight]["weight"])
    #     npdist.all_gather(grads_tp[weight]["weight"], x, axis=-1, comm=tp_comm)
    #     np.testing.assert_allclose(grads[weight]["weight"], x, atol=1e-5)
