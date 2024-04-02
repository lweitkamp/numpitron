import numpy as np
from mpi4py import MPI


def reduce(
    tensor: np.ndarray,
    dst: int = 0,
    op: MPI.Op = MPI.SUM,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Reduce tensor across all processes and broadcast the result
    back to a single process.

    Args:
        tensor (np.ndarray): NumPy array.
        dst (int): Rank on which we gather the reduction.
        op (MPI.Op): Operation to reduce the tensor.
        comm (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    if comm.Get_rank() == dst:
        comm.Reduce(MPI.IN_PLACE, tensor, op=op, root=dst)
    else:
        comm.Reduce(tensor, None, op=op, root=dst)
