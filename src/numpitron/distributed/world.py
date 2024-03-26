import numpy as np
from mpi4py import MPI


MPI_COMM = MPI.COMM_WORLD


def get_rank() -> int:
    """Return the rank of the current process."""
    return MPI_COMM.Get_rank()


def world_size() -> int:
    """Return the world size, i.e. the number of parallel processes."""
    return MPI_COMM.Get_size()


def init_process_grid(grid: tuple[int, ...] | int) -> None:
    """Initialize the process grid, defining the position of each
    process in the grid hierarchy."""
    assert np.prod(grid) == world_size()
    
    if isinstance(grid, int):
        grid = (grid, 1, 1)

    grid = grid + (1,) * (3 - len(grid))

    assert (
        len(grid) == 3
    ), f"Grid can only have up to three dimensions, found {len(grid)}"


def assert_divisible(dim: int) -> None:
    """A simple assert to check that whatever the size of the
    dimension is, it can be equally divided among devices."""
    assert (
        dim % world_size() == 0
    ), f"Cannot divide the dimension {dim} amongst {world_size()} devices."
