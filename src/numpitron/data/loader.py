from pathlib import Path

import numpy as np
from numpy.random import Generator
import tqdm


from numpitron import distributed as npdist


class DataLoader:
    def __init__(
        self,
        dataset_path: str | Path,
        seq_len: int,
        batch_size: int,
        rng: Generator,
    ):
        """

        Args:
            dataset_path (str | Path): Path to tokenized data.
            seq_len (int): Maximum sequence length per batch.
            batch_size (int): Number of batches to sample.
            rng (Generator): NumPy random number generator.
        """
        # Convert to numpy integers.
        self.dataset_path = dataset_path
        self.rng = rng

        # Calculate how quickly we go through one epoch.
        _data = np.memmap(self.dataset_path, dtype=np.uint16, mode="r")
        self.batches_per_epoch = len(_data) // (seq_len * batch_size)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_range = np.arange(seq_len)

    def iter_epoch(self):
        """Go through one iteration of the dataset.

        Returns:
            An iterable dataset.
        """

        def _iter_epoch():
            batch_idx = 0
            data = np.memmap(self.dataset_path, dtype=np.uint16, mode="r")
            while batch_idx <= self.batches_per_epoch:
                start_idx = self.rng.integers(
                    0, len(data) - 1 - self.seq_len, size=self.batch_size
                )[:, None]
                inputs = data[start_idx + self.seq_range]
                labels = data[start_idx + 1 + self.seq_range]

                if npdist.data_parallel_size() > 1:
                    local_batch_size = self.batch_size // npdist.data_parallel_size()
                    inputs_local = np.empty(
                        (local_batch_size, self.seq_len), dtype=inputs.dtype
                    )
                    labels_local = np.empty(
                        (local_batch_size, self.seq_len), dtype=labels.dtype
                    )

                    npdist.scatter(
                        inputs,
                        inputs_local,
                        axis=0,
                        src=0,
                        group=npdist.data_parallel_group(),
                    )
                    npdist.scatter(
                        labels,
                        labels_local,
                        axis=0,
                        src=0,
                        group=npdist.data_parallel_group(),
                    )
                    inputs = inputs_local
                    labels = labels_local

                yield inputs, labels

                batch_idx += 1

        pbar = tqdm.tqdm(
            _iter_epoch(),
            leave=False,
            total=self.batches_per_epoch,
            disable=npdist.world_rank() != 0,
        )
        return pbar
