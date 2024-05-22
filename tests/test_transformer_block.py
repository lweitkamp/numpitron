from copy import deepcopy

import numpy as np
import pytest

from numpitron import distributed as npdist
from numpitron.nn import TransformerBlock


def test_to_from_dict():
    b, s, d, n = 2, 16, 32, 4

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d)).astype(np.float32)
    transformer_block = TransformerBlock(d, n, rng=rng)
    outputs = transformer_block(inputs)

    transformer_block_dict = transformer_block.to_dict()
    transformer_block_from_dict = TransformerBlock.from_dict(transformer_block_dict)
    outputs_from_dict = transformer_block_from_dict(inputs)

    np.testing.assert_allclose(outputs, outputs_from_dict)


def test_from_savefile(tmp_path):
    b, s, d, n = 2, 16, 32, 4

    rng = np.random.default_rng(42)
    save_path = tmp_path / "model.npy"

    inputs = rng.random((b, s, d)).astype(np.float32)
    transformer_block = TransformerBlock(d, n, rng=rng)
    outputs = transformer_block(inputs)

    np.save(save_path, transformer_block.to_dict())
    transformer_block_dict = np.load(save_path, allow_pickle=True)[()]
    transformer_block_from_dict = TransformerBlock.from_dict(transformer_block_dict)
    outputs_from_dict = transformer_block_from_dict(inputs)

    np.testing.assert_allclose(outputs, outputs_from_dict)


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_tensor_parallel():
    b, s, d, n = 2, 16, 32, 4

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, d)).astype(np.float32)
    transformer_block = TransformerBlock(d, n, rng=rng)
    single = deepcopy(transformer_block)

    transformer_block.scatter()

    outputs = transformer_block(inputs)
    outputs_single = single(inputs)

    np.testing.assert_allclose(outputs, outputs_single)
