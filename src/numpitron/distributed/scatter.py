import numpy as np
from mpi4py import MPI

from numpitron.distributed.world import world_size


MPI_COMM = MPI.COMM_WORLD


def scatter(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    axis: int,
    src: int = 0,
) -> None:
    """We scatter the source tensor along an axis and the scattered result
    will be collected in the destination tensor for each process.

    Args:
        source_tensor (np.ndarray): Tensor to scatter along axis.
        destination_tensor (np.ndarray): Tensor from each process to
            collect results.
        axis (int): axis to split source_tensor and scatter the results with.
        src (int): Rank from which we scatter the tensor.
    """
    scatter_list = np.split(source_tensor, world_size(), axis=axis)
    scatter_list = np.concatenate([np.ravel(x) for x in scatter_list])
    MPI_COMM.Scatterv(scatter_list, destination_tensor, root=src)
