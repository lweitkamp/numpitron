import numpy as np
from mpi4py import MPI


def all_reduce(
    tensor: np.ndarray,
    op: MPI.Op = MPI.SUM,
    group: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Reduce tensor across all processes and broadcast the result
    back to all processes.

    Args:
        tensor (np.ndarray): NumPy array.
        op (MPI.Op): Operation to reduce the tensor.
        group (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    group.Allreduce(MPI.IN_PLACE, tensor, op=op)
