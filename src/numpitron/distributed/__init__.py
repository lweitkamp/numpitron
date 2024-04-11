# flake8: noqa
from dataclasses import dataclass, replace, field

from numpitron.distributed.all_gather import all_gather
from numpitron.distributed.all_reduce import all_reduce
from numpitron.distributed.all_to_all import all_to_all
from numpitron.distributed.broadcast import broadcast
from numpitron.distributed.gather import gather
from numpitron.distributed.recv import recv
from numpitron.distributed.reduce_scatter import reduce_scatter
from numpitron.distributed.reduce import reduce
from numpitron.distributed.scatter import scatter
from numpitron.distributed.send import send

import numpy as np
from mpi4py import MPI


@dataclass
class ParallelState:
    """A parallel state data class. Holds all communication groups."""

    world_group: MPI.Group = field(default_factory=lambda: MPI.COMM_WORLD)
    tensor_parallel_group: MPI.Group = None
    pipeline_parallel_group: MPI.Group = None
    data_parallel_group: MPI.Group = None
    model_parallel_group: MPI.Group = None


# The default parallel state only has the world group active.
PARALLEL_STATE = ParallelState()


def world_size() -> int:
    """Returns the size of the world group (all possible ranks)."""
    return PARALLEL_STATE.world_group.Get_size()


def world_rank() -> int:
    """Returns the rank of the current process in the world group."""
    return PARALLEL_STATE.world_group.Get_rank()


def tensor_parallel_size() -> int:
    """Returns the size of current rank's tensor parallel group."""
    assert (
        PARALLEL_STATE.tensor_parallel_group is not None
    ), "Tensor Parallel group is not initiated."
    return PARALLEL_STATE.tensor_parallel_group.Get_size()


def tensor_parallel_rank() -> int:
    """Returns the rank of the current process in the tensor parallel group."""
    assert (
        PARALLEL_STATE.tensor_parallel_group is not None
    ), "Tensor Parallel group is not initiated."
    return PARALLEL_STATE.tensor_parallel_group.Get_rank()


def tensor_parallel_group() -> MPI.Group:
    """Returns the tensor parallel group related to the current rank."""
    assert (
        PARALLEL_STATE.tensor_parallel_group is not None
    ), "Tensor Parallel group is not initiated."
    return PARALLEL_STATE.tensor_parallel_group


def pipeline_parallel_size() -> int:
    """Returns the size of current rank's pipeline parallel group."""
    assert (
        PARALLEL_STATE.pipeline_parallel_group is not None
    ), "Pipeline Parallel group is not initiated."
    return PARALLEL_STATE.pipeline_parallel_group.Get_size()


def pipeline_parallel_rank() -> int:
    """Returns the rank of the current process in the pipeline parallel group."""
    assert (
        PARALLEL_STATE.pipeline_parallel_group is not None
    ), "Pipeline Parallel group is not initiated."
    return PARALLEL_STATE.pipeline_parallel_group.Get_rank()


def pipeline_parallel_group() -> MPI.Group:
    """Returns the pipeline parallel group related to the current rank."""
    assert (
        PARALLEL_STATE.pipeline_parallel_group is not None
    ), "Pipeline Parallel group is not initiated."
    return PARALLEL_STATE.pipeline_parallel_group


def data_parallel_size() -> int:
    """Returns the size of current rank's data parallel group."""
    assert (
        PARALLEL_STATE.data_parallel_group is not None
    ), "Data Parallel group is not initiated."
    return PARALLEL_STATE.data_parallel_group.Get_size()


def data_parallel_rank() -> int:
    """Returns the rank of the current process in the data parallel group."""
    assert (
        PARALLEL_STATE.data_parallel_group is not None
    ), "Data Parallel group is not initiated."
    return PARALLEL_STATE.data_parallel_group.Get_rank()


def data_parallel_group() -> MPI.Group:
    """Returns the daata parallel group related to the current rank."""
    assert (
        PARALLEL_STATE.data_parallel_group is not None
    ), "Data Parallel group is not initiated."
    return PARALLEL_STATE.data_parallel_group


def add_group(all_groups_ranks: np.ndarray):
    world_comm = MPI.Comm(MPI.COMM_WORLD)
    for group_ranks in all_groups_ranks:
        sorted_ranks = tuple(sorted(group_ranks))
        if world_rank() in sorted_ranks:
            return world_comm.Create(world_comm.group.Incl(ranks=sorted_ranks))
    return None


def init(tp_size: int = 1, pp_size: int = 1, dp_size: int = 1) -> None:
    """Add tensor, pipeline, and data parallel groups to PARALLEL_STATE.

    Args:
        tp_size (int): The number of devices for one tensor parallel group.
        pp_size (int): The number of devices for one pipeline parallel group.
        dp_size (int): The number of devices for one data parallel group.
    """
    # Define the number of GPUs it takes to load one full model.
    assert tp_size * pp_size * dp_size == world_size(), (
        "The total world size must match the product of parallelization "
        f"strategies. {world_size()=}, {tp_size=}, {pp_size=}, {dp_size=}"
    )

    # Do init all groups.
    ranks = np.arange(0, world_size()).reshape(pp_size, dp_size, tp_size)

    global PARALLEL_STATE
    PARALLEL_STATE = replace(
        PARALLEL_STATE,
        tensor_parallel_group=add_group(
            ranks.transpose((0, 1, 2)).reshape(-1, tp_size)
        ),
        pipeline_parallel_group=add_group(
            ranks.transpose((1, 2, 0)).reshape(-1, pp_size)
        ),
        data_parallel_group=add_group(
            ranks.transpose((2, 0, 1)).reshape(-1, dp_size)
        ),
        model_parallel_group=add_group(
            [ranks[:, dp_rank, :].reshape(-1) for dp_rank in range(dp_size)]
        ),
    )
