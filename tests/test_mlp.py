import pytest
import numpy as np

from numpitron.nn import MLP
from numpitron import distributed as npdist


npdist.init(tp_size=npdist.world_size())


def test_linear():
    b, s, d = 32, 64, 128

    inputs = np.ones((b, s, d))
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
    row_linear = Linear(d, d, init_fn=lambda shape: np.ones(shape), weight_shard_axis=1, bias_shard_axis=0)
    row_linear.scatter()

    outputs = row_linear(inputs)

    np.testing.assert_allclose(
        outputs.sum(), b * s * d * (d // size) + b * s * (d // size)
    )
    assert outputs.shape == (b, s, d // size)
