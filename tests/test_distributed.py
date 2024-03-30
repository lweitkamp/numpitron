import numpy as np
import pytest

import numpitron.distributed as npdist


RANK = npdist.get_rank()
WORLD_SIZE = npdist.world_size()

assert WORLD_SIZE > 1, "Run these unit tests at least with n>1 processes."


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_all_gather(
    batch_size: int,
    seq_len: int,
) -> None:
    destination_tensor = np.zeros((batch_size, seq_len, WORLD_SIZE))
    source_tensor = np.zeros((batch_size, seq_len, 1)) + RANK

    npdist.all_gather(source_tensor, destination_tensor, axis=-1)

    assert set(np.unique(destination_tensor)) == set(range(WORLD_SIZE))


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_all_reduce(
    batch_size: int,
    seq_len: int,
) -> None:
    tensor = np.zeros((batch_size, seq_len, WORLD_SIZE))
    tensor[..., RANK] = 1.0

    npdist.all_reduce(tensor)

    np.testing.assert_equal(tensor.sum(), batch_size * seq_len * WORLD_SIZE)


def test_all_to_all() -> None:
    raise Exception()


def test_barrier() -> None:
    raise Exception()


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(1, 2, 4), (2, 4, 8)],
)
def test_broadcast(
    batch_size: int,
    seq_len: int,
    d_model: int,
) -> None:
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


def test_gather() -> None:
    raise Exception()


def rest_reduce_scatter() -> None:
    raise Exception()


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_reduce(
    batch_size: int,
    seq_len: int,
) -> None:
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
def test_scatter(
    batch_size: int,
    seq_len: int,
) -> None:
    source_tensor = np.zeros((batch_size, seq_len, WORLD_SIZE))
    destination_tensor = np.zeros((batch_size, seq_len, 1))

    if RANK == 0:
        source_tensor = source_tensor + 1.0

    npdist.scatter(source_tensor, destination_tensor, axis=-1)

    np.testing.assert_equal(destination_tensor.sum(), batch_size * seq_len)


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(1, 2, 4), (2, 4, 8)],
)
def test_send_recv(
    batch_size: int,
    seq_len: int,
    d_model: int,
) -> None:
    source_tensor = np.ones((batch_size, seq_len, d_model))
    destination_tensor = np.zeros_like(source_tensor)

    if RANK == 0:
        npdist.send(source_tensor, dst=1)

    if RANK == 1:
        npdist.recv(destination_tensor, src=0)
        np.testing.assert_equal(source_tensor, destination_tensor)

    if RANK == 0:
        np.testing.assert_equal(destination_tensor, np.zeros_like(source_tensor))