import numpy as np
from mpi4py import MPI


def gather(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    dst: int = 0,
    axis: int = -1,
    group: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """Gather source tensor from all processes and store it in destination
    tensor.

    Args:
        source_tensor (np.ndarray): Tensor to store result in.
        destination_tensor (np.ndarray): Tensor to store result in.
        dst (int): Rank on which we gather the data.
        axis (int): The axis on which the tensor needs to be concatenated.
        group (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    receiving_buffer = np.empty(
        np.prod(destination_tensor.shape), dtype=source_tensor.dtype
    )
    group.Gatherv(source_tensor, receiving_buffer, root=dst)

    if group.Get_rank() == dst:
        receiving_buffer = np.split(receiving_buffer, group.Get_size(), 0)
        receiving_buffer = np.concatenate(
            [x.reshape(source_tensor.shape) for x in receiving_buffer],
            axis=axis,
        )
        np.copyto(destination_tensor, receiving_buffer)
