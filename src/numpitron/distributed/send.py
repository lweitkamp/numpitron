import numpy as np
from mpi4py import MPI


MPI_COMM = MPI.COMM_WORLD


def send(
    source_tensor: np.ndarray,
    dst: int,
    tag: int = 0,
) -> None:
    """Send a tensor to destination. Optional tag flag to denote the type of
    message to send.

    Args:
        source_tensor (np.ndarray): Tensor to send.
        dst (int): Destination to send tensor to.
        tag (int): Tag to indicate type of message (if needed).
    """
    MPI_COMM.Send(source_tensor, dst, tag=tag)
