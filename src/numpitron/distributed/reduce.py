import numpy as np
from mpi4py import MPI


def reduce(
    tensor: np.ndarray,
    dst: int = 0,
    op: str = "sum",
    group: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Reduce tensor across all processes and broadcast the result
    back to a single process.

    Args:
        tensor (np.ndarray): NumPy array.
        dst (int): Rank on which we gather the reduction.
        op (MPI.Op): Operation to reduce the tensor.
        group (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    op = {
        "sum": MPI.SUM,
        "max": MPI.MAX,
    }[op]

    if group.Get_rank() == dst:
        group.Reduce(MPI.IN_PLACE, tensor, op=op, root=dst)
    else:
        group.Reduce(tensor, None, op=op, root=dst)
