import numpy as np
from mpi4py import MPI


def all_gather(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    axis: int = -1,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Gather source tensors from each process and collect it in the
    destination tensor.

    MPI sends a contiguous stream of bytes from each process. To ensure
    the expected shape is returned in destination tensor, we collect
    the contiguous stream per process and reshape each accordingly.

    Args:
        source_tensor (np.ndarray): Source tensor for each process.
        destination_tensor (np.ndarray): Tensor to gather the results.
        axis (int): The axis on which the tensor needs to be concatenated.
        comm (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    receiving_buffer = np.empty(
        np.prod(destination_tensor.shape), dtype=source_tensor.dtype
    )
    comm.Allgather(source_tensor, receiving_buffer)
    receiving_buffer = np.split(receiving_buffer, comm.Get_size(), axis)
    receiving_buffer = np.concatenate(
        [x.reshape(source_tensor.shape) for x in receiving_buffer],
        axis=axis,
    )
    np.copyto(destination_tensor, receiving_buffer)
