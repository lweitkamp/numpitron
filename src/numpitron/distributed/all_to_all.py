import numpy as np
from mpi4py import MPI


MPI_COMM = MPI.COMM_WORLD

def all_to_all(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
) -> None:
    """Scatter the source tensor to all other ranks and gather whatever
    the other ranks scattered.

    Args:
        source_tensor (np.ndarray): Source tensor for each process.
        destination_tensor (np.ndarray): Tensor to gather the results.
    """
    MPI_COMM.Alltoallv(source_tensor, destination_tensor)
