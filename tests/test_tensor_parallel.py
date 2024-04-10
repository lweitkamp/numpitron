"""Every tests follows the following logic:

1. Initiate parallel and non-parallel layer.
2. 'Scatter' non-parallel weights to parallel layer.
3. Run a forward + backward pass with both.
4. Check outputs + the 'Gather' results.
"""

import numpy as np
import pytest
from numpy.random import Generator

from numpitron import nn, distributed as npdist
from numpitron.tensor_parallel import TensorParallelMLP, TensorParallelInputEmbedding

npdist.init(tp_size=npdist.world_size())


TEST_SHAPES = [
    (1, 1, 8),
    (3, 1, 8),
    (1, 3, 8),
    (3, 3, 8),
]


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng(42)


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,vocab_size", [shape + (20,) for shape in TEST_SHAPES]
)
def test_embedding(
    batch_size: int,
    seq_len: int,
    d_model: int,
    vocab_size: int,
    rng: Generator,
):
    inputs = rng.integers((batch_size, seq_len, vocab_size))

    embedding = nn.InputEmbedding(d_model, vocab_size)
    embedding_tp = TensorParallelInputEmbedding(d_model, vocab_size)
    params = embedding.init_params(rng=rng)
    params_tp = embedding_tp.init_params(rng=rng)

    npdist.scatter(
        params["embedding"],
        params_tp["embedding"],
        axis=1,
        group=npdist.tensor_parallel_group(),
    )

    ctx, out = embedding(params, inputs)
    ctx_tp, out_tp = embedding_tp(params_tp, inputs)

    grads, d_out = embedding.backward(ctx, np.ones_like(out))
    grads_tp, d_out_tp = embedding_tp.backward(ctx_tp, np.ones_like(out_tp))

    g = np.zeros_like(grads["embedding"])
    npdist.all_gather(
        grads_tp["embedding"], g, axis=-1, group=npdist.tensor_parallel_group()
    )

    np.testing.assert_allclose(out, out_tp, atol=1e-5)
    np.testing.assert_allclose(d_out, d_out_tp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g, grads["embedding"], atol=1e-5)


@pytest.mark.parametrize("batch_size,seq_len,d_model", TEST_SHAPES)
def test_mlp(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)

    mlp = nn.MLP(d_model, d_model * 4)
    mlp_tp = TensorParallelMLP(d_model, d_model * 4)
    params = mlp.init_params(rng=rng)
    params_tp = mlp_tp.init_params(rng=rng)

    for i, weight in enumerate(["Linear1", "Linear2"]):
        npdist.scatter(
            params[weight]["weight"],
            params_tp[weight]["weight"],
            axis=1 if i == 0 else 0,
            group=npdist.tensor_parallel_group(),
        )
        if i == 0:
            npdist.scatter(
                params[weight]["bias"],
                params_tp[weight]["bias"],
                axis=0,
                group=npdist.tensor_parallel_group(),
            )

    ctx, out = mlp(params, inputs)
    ctx_tp, out_tp = mlp_tp(params_tp, inputs)

    grads, d_out = mlp.backward(ctx, np.ones_like(out))
    grads_tp, d_out_tp = mlp_tp.backward(ctx_tp, np.ones_like(out_tp))

    np.testing.assert_allclose(out, out_tp, atol=1e-5)
    np.testing.assert_allclose(d_out, d_out_tp, rtol=1e-5, atol=1e-5)

    for i, weight in enumerate(["Linear1", "Linear2"]):
        w = np.zeros_like(grads[weight]["weight"])
        npdist.all_gather(
            grads_tp[weight]["weight"],
            w,
            axis=-1 if i == 0 else 0,
            group=npdist.tensor_parallel_group(),
        )
        np.testing.assert_allclose(grads[weight]["weight"], w, atol=1e-5)

        if i == 0:
            b = np.zeros_like(grads[weight]["bias"])
            npdist.all_gather(
                grads_tp[weight]["bias"],
                b,
                axis=0,
                group=npdist.tensor_parallel_group(),
            )
            np.testing.assert_allclose(grads[weight]["bias"], b, atol=1e-5)
