import numpy as np
from mpi4py import MPI


def all_reduce(
    tensor: np.ndarray,
    op: MPI.Op = MPI.SUM,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Reduce tensor across all processes and broadcast the result
    back to all processes.

    Args:
        tensor (np.ndarray): NumPy array.
        op (MPI.Op): Operation to reduce the tensor.
        comm (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    comm.Allreduce(MPI.IN_PLACE, tensor, op=op)
