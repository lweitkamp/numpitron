import numpy as np
from mpi4py import MPI


def all_gather(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    axis: int = -1,
    group: MPI.Intracomm = MPI.COMM_WORLD,
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
        group (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    # In order to deal with possible non-uniform splits, we need to pass
    # the shapes and displacements to the allgatherv function.
    gather_list = np.array_split(destination_tensor, group.Get_size(), axis=axis)
    gather_shapes = [x.shape for x in gather_list]
    shapes = [np.array(np.prod(x.shape)) for x in gather_list]
    displ = [sum(shapes[:p]) for p in range(len(shapes))]

    receiving_buffer = np.empty(
        np.prod(destination_tensor.shape), dtype=source_tensor.dtype
    )
    group.Allgatherv(source_tensor, [receiving_buffer, shapes, displ, MPI.FLOAT])

    # Now to ensure everything is back in the correct place.
    receiving_buffer = np.concatenate(
        [
            x.reshape(shape)
            for x, shape in zip(np.split(receiving_buffer, displ[1:]), gather_shapes)
        ],
        axis=axis,
    )
    np.copyto(destination_tensor, receiving_buffer)
