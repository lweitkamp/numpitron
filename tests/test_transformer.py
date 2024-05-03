import numpy as np

from numpitron import distributed as npdist
from numpitron.nn import Transformer

npdist.init(tp_size=npdist.world_size())


def test_to_from_dict():
    b, s, d, n, l, v = 2, 16, 32, 4, 3, 32

    rng = np.random.default_rng(42)

    inputs = rng.integers(0, v, (b, s))
    transformer = Transformer(d, n, l, v, s, rng=rng)
    outputs = transformer(inputs)

    transformer_dict = transformer.to_dict()
    transformer_from_dict = Transformer.from_dict(transformer_dict)
    outputs_from_dict = transformer_from_dict(inputs)

    np.testing.assert_allclose(outputs, outputs_from_dict)
