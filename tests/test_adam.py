import numpy as np

from numpitron import distributed as npdist
from numpitron.nn import Transformer
from numpitron.optimizer import Adam

npdist.init(tp_size=npdist.world_size())


def test_forward_backward():
    b, s, d, n, l, v = 2, 16, 32, 4, 3, 32

    rng = np.random.default_rng(42)

    inputs = rng.integers(0, v, (b, s))
    transformer = Transformer(d, n, l, v, s, rng=rng)
    optimizer = Adam(transformer, 1e-4)

    outputs = transformer(inputs)
    transformer.backward(np.ones_like(outputs))
    optimizer.step()
