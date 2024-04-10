import numpy as np
from mpi4py import MPI


def broadcast(
    tensor: np.ndarray,
    src: int = 0,
    group: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Broadcast tensor to all devices.

    Args:
        tensor (np.ndarray): NumPy array.
        src (int): Source rank from which to broadcast.
        group (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    np.copyto(tensor, group.bcast(tensor, root=src))
