import numpy as np
from mpi4py import MPI


def recv(
    destination_tensor: np.ndarray,
    src: int,
    tag: int = 0,
    status = None,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Receive a tensor from source and store in destination tensor.

    Args:
        destination_tensor (np.ndarray): Tensor to store result in.
        src (int): Source of the tensor.
        tag (int): Tag to indicate type of message (if needed).
        comm (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    comm.Recv(destination_tensor, src, tag=tag, status=status)
