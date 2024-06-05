import numpy as np
from mpi4py import MPI


def scatter(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    axis: int,
    src: int = 0,
    group: MPI.Intracomm = MPI.COMM_WORLD,
) -> None:
    """We scatter the source tensor along an axis and the scattered result
    will be collected in the destination tensor for each process.

    Args:
        source_tensor (np.ndarray): Tensor to scatter along axis.
        destination_tensor (np.ndarray): Tensor from each process to
            collect results.
        axis (int): axis to split source_tensor and scatter the results with.
        src (int): Rank from which we scatter the tensor.
        group (MPI.Intracomm): MPI Communicator. Defaults to WORLD.
    """
    # In order to deal with possible non-uniform splits, we need to pass
    # the shapes and displacements to the scatterv function.
    scatter_list = np.array_split(source_tensor, group.Get_size(), axis=axis)
    shapes = [np.array(np.prod(x.shape)) for x in scatter_list]
    displ = [sum(shapes[:p]) for p in range(len(shapes))]

    scatter_list = np.concatenate([x.reshape(-1) for x in scatter_list])

    DTYPES = {"float32": MPI.FLOAT, "uint16": MPI.UINT16_T}
    group.Scatterv(
        [scatter_list, shapes, displ, DTYPES[source_tensor.dtype.name]],
        destination_tensor,
        root=src,
    )
