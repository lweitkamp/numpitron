# flake8: noqa
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


class WorldCommunicator:
    """Distributed communication handler. This class only initiates the world
    communication."""
    def __init__(self):
        self.world_comm = MPI.COMM_WORLD
        self.world_size = self.world_comm.Get_size()
        self.world_rank = self.world_comm.Get_rank()


class ParallelCommunicator(WorldCommunicator):
    """Distributed communication handler. Most code is taken from Nanotron:
    https://github.com/huggingface/nanotron/blob/main/src/nanotron/parallel/context.py.

    All communications are mediated through an instantiation of this object.
    """

    def __init__(self, tp_size: int = 1, pp_size: int = 1, dp_size: int = 1):
        """
        Args:
            tp_size (int): Tensor Parallel size.
            pp_size (int): Pipeline Parallel size.
            dp_size (int): Data Parallel size.
            debug (bool): Testing flag to ignore 3D parallelization.
        """
        super().__init__()

        assert tp_size * pp_size * dp_size == self.world_size, (
            "Total world size needs to be equal to 3D parallel size."
            f" Found: {self.world_size=} != {tp_size=} * {pp_size=} * {dp_size=}"
        )

        ranks = np.arange(self.world_size).reshape((dp_size, pp_size, tp_size))

        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.rank_to_group = {}

        self.tp_comm = self.create_comm(
            ranks.transpose((0, 1, 2)).reshape((-1, tp_size))
        )
        self.pp_comm = self.create_comm(
            ranks.transpose((1, 2, 0)).reshape((-1, pp_size))
        )
        self.dp_comm = self.create_comm(
            ranks.transpose((2, 0, 1)).reshape((-1, dp_size))
        )

    def create_comm(self, ranks: np.ndarray) -> MPI.Comm:
        """Create an MPI communicator group."""
        self.world_comm.Barrier()

        new_group_containing_rank = None
        for group_ranks in ranks:
            sorted_ranks = tuple(sorted(group_ranks))

            # add new group to `world_ranks_to_pg`
            if sorted_ranks not in self.rank_to_group:
                new_group = MPI.Comm(self.world_comm).Create(
                    self.world_comm.group.Incl(ranks=group_ranks)
                )
                self.rank_to_group[sorted_ranks] = new_group
            else:
                new_group = self.rank_to_group[sorted_ranks]

            if self.world_rank in sorted_ranks:
                new_group_containing_rank = new_group

        self.world_comm.Barrier()
        return new_group_containing_rank
