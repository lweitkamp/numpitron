import numpy as np
from mpi4py import MPI


def send(
    source_tensor: np.ndarray,
    dst: int,
    tag: int = 0,
    group: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Send a tensor to destination. Optional tag flag to denote the type of
    message to send.

    Args:
        source_tensor (np.ndarray): Tensor to send.
        dst (int): Destination to send tensor to.
        tag (int): Tag to indicate type of message (if needed).
        group (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    group.Send(source_tensor, dst, tag=tag)
