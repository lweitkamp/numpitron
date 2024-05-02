import numpy as np

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
