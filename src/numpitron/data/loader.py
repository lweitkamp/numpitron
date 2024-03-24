""""""
from pathlib import Path

import numpy as np


class DataLoader:
    def __init__(
        self,
        dataset_path: str | Path,
        rng,
        seq_len: int,
        batch_size: int,
    ):
        """

        Args:
            dataset_path (str | Path): Path to tokenized data.
        """
        # Convert to numpy integers.
        self.dataset_path = dataset_path
        self.rng = rng

        # Calculate how quickly we go through one epoch.
        _data = np.memmap(self.dataset_path, dtype=np.uint16, mode='r')
        self.batches_per_epoch = len(_data) // (seq_len * batch_size)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_range = np.arange(seq_len)

    def iter_epoch(self):
        batch_idx = 0
        data = np.memmap(self.dataset_path, dtype=np.uint16, mode='r')
        while batch_idx <= self.batches_per_epoch:
            start_idx = self.rng.integers(
                0, len(data) - 1 - self.seq_len, size=self.batch_size
            )[:, None]
            inputs = data[start_idx + self.seq_range]
            labels = data[start_idx + 1 + self.seq_range]
            yield inputs, labels

            batch_idx += 1
