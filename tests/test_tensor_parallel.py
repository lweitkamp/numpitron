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
from numpitron.tensor_parallel import TensorParallelMLP

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
def test_mlp(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)

    mlp = nn.MLP(d_model, d_model * 4)
    mlp_tp = TensorParallelMLP(d_model, d_model * 4, communicator=comm)
    params = mlp.init_params(rng=rng)
    params_tp = mlp_tp.init_params(rng=rng)

    npdist.scatter(
        params["Linear1"]["weight"],
        params_tp["ColumnLinear"]["weight"],
        axis=1,
        comm=comm.tp_comm,
    )
    npdist.scatter(
        params["Linear1"]["bias"],
        params_tp["ColumnLinear"]["bias"],
        axis=0,
        comm=comm.tp_comm,
    )
    npdist.scatter(
        params["Linear2"]["weight"],
        params_tp["RowLinear"]["weight"],
        axis=0,
        comm=comm.tp_comm,
    )

    ctx, out = mlp(params, inputs)
    ctx_tp, out_tp = mlp_tp(params_tp, inputs)

    grads, d_out = mlp.backward(ctx, np.ones_like(out))
    grads_tp, d_out_tp = mlp_tp.backward(ctx_tp, np.ones_like(out_tp))

    clw = np.zeros_like(grads["Linear1"]["weight"])
    clb = np.zeros_like(grads["Linear1"]["bias"])
    rlw = np.zeros_like(grads["Linear2"]["weight"])

    npdist.all_gather(
        grads_tp["ColumnLinear"]["weight"], clw, axis=-1, comm=comm.tp_comm
    )
    npdist.all_gather(grads_tp["ColumnLinear"]["bias"], clb, axis=0, comm=comm.tp_comm)
    npdist.all_gather(grads_tp["RowLinear"]["weight"], rlw, axis=0, comm=comm.tp_comm)

    np.testing.assert_allclose(d_out, d_out_tp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out, out_tp, atol=1e-5)
    np.testing.assert_allclose(grads["Linear1"]["weight"], clw, atol=1e-5)
    np.testing.assert_allclose(grads["Linear1"]["bias"], clb, atol=1e-5)
    np.testing.assert_allclose(grads["Linear2"]["weight"], rlw, atol=1e-5)
