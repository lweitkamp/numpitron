import numpy as np
from mpi4py import MPI


def all_to_all(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Scatter the source tensor to all other ranks and gather whatever
    the other ranks scattered.

    Args:
        source_tensor (np.ndarray): Source tensor for each process.
        destination_tensor (np.ndarray): Tensor to gather the results.
        comm (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    comm.Alltoallv(source_tensor, destination_tensor)
