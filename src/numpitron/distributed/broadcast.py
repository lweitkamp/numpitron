import numpy as np
from mpi4py import MPI


MPI_COMM = MPI.COMM_WORLD


def broadcast(
    tensor: np.ndarray,
    src: int = 0,
) -> None:
    """Broadcast tensor to all devices.

    Args:
        tensor (np.ndarray): NumPy array.
        src (int): Source rank from which to broadcast.
    """
    np.copyto(tensor, MPI_COMM.bcast(tensor, root=src))
