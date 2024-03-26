import numpy as np
from mpi4py import MPI

from numpitron.distributed.reduce import reduce
from numpitron.distributed.scatter import scatter


MPI_COMM = MPI.COMM_WORLD


def reduce_scatter(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    op: MPI.Op = MPI.SUM,
) -> None:
    """Reduce source tensor to root process and scatter the reduction
    back to all processes.

    Again the issue with contiguous data streams - but I could not
    find a workaround for MPI_COMM.Reduce_scatter so I resorted to using
    the `reduce` and `scatter` operations I defined earlier.

    Args:
        source_tensor (np.ndarray): Source tensor for each process.
        destination_tensor (np.ndarray): Tensor to gather the results.
        op (MPI.Op): Operation to reduce the tensor.
    """
    # Maybe soon:
    # MPI_COMM.Reduce_scatter(source_tensor, destination_tensor, op=op)
    reduce(source_tensor, dst=0, op=op)
    scatter(source_tensor, destination_tensor, axis=-1, src=0)
