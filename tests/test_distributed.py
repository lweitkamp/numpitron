import numpy as np
import pytest

import numpitron.distributed as npdist


RANK = npdist.get_rank()
WORLD_SIZE = npdist.world_size()


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(1, 2, 4), (2, 4, 8)],
)
def test_broadcast(
    batch_size: int,
    seq_len: int,
    d_model: int,
) -> None:
    """Create a tensor on root, broadcast it, assert
    all RANKs have the same tensor afterwards."""
    expected_tensor = np.arange((batch_size * seq_len * d_model)).reshape(
        batch_size, seq_len, d_model
    )

    tensor = (
        np.empty((batch_size, seq_len, d_model))
        if RANK != 0
        else np.copy(expected_tensor)
    )

    npdist.broadcast(tensor)

    np.testing.assert_array_equal(tensor, expected_tensor)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_reduce(
    batch_size: int,
    seq_len: int,
) -> None:
    """Create an empty tensor and fill specific RANK indices with ones.
    After reduction, only the root RANK should have a sum equal to
    the entire tensor shape, all others should have a sum equal to whatever
    they filled in."""
    tensor = np.zeros((batch_size, seq_len, WORLD_SIZE))
    tensor[..., RANK] = 1.0

    npdist.reduce(tensor)

    if RANK == 0:
        np.testing.assert_equal(tensor.sum(), batch_size * seq_len * WORLD_SIZE)
    else:
        np.testing.assert_equal(tensor.sum(), batch_size * seq_len)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_all_reduce(
    batch_size: int,
    seq_len: int,
) -> None:
    """Create an empty tensor and fill specific RANK indices with ones.
    After all-reduce, all RANKs should have a sum equal to
    the entire tensor shape."""
    tensor = np.zeros((batch_size, seq_len, WORLD_SIZE))
    tensor[..., RANK] = 1.0

    npdist.all_reduce(tensor)

    np.testing.assert_equal(tensor.sum(), batch_size * seq_len * WORLD_SIZE)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_scatter(
    batch_size: int,
    seq_len: int,
) -> None:
    """Create a zeros tensor and fill it with ones on root. Scatter it to
    all processes and ensure the sum is expected."""
    source_tensor = np.zeros((batch_size, seq_len, WORLD_SIZE))
    destination_tensor = np.zeros((batch_size, seq_len, 1))

    if RANK == 0:
        source_tensor = source_tensor + 1.0

    npdist.scatter(source_tensor, destination_tensor, axis=-1)

    np.testing.assert_equal(destination_tensor.sum(), batch_size * seq_len)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_all_gather(
    batch_size: int,
    seq_len: int,
) -> None:
    """Each process creates a tensor with their RANK as value. We all-gather
    the result to all processes. To test if the gather was successful, each
    slice sent by a process should have unique values."""
    destination_tensor = np.zeros((batch_size, seq_len, WORLD_SIZE))
    source_tensor = np.zeros((batch_size, seq_len, 1)) + RANK

    npdist.all_gather(source_tensor, destination_tensor, axis=-1)

    assert set(np.unique(destination_tensor)) == set(range(WORLD_SIZE))
