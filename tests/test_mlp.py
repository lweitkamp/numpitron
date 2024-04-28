import pytest
import numpy as np

from numpitron.nn import MLP
from numpitron import distributed as npdist


npdist.init(tp_size=npdist.world_size())


def test_mlp():
    b, s, d = 32, 64, 128

    inputs = np.ones((b, s, d))
    mlp = MLP(d, d * 4, d)
    outputs = mlp(inputs)


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_tensor_parallel_mlp():
    b, s, d = 32, 64, 128

    inputs = np.ones((b, s, d))
    mlp = MLP(d, d * 4, d)
    mlp_tp = MLP(d, d * 4, d)

    mlp_tp.scatter()

    out = mlp(inputs)
    out_tp = mlp_tp(inputs)

    if npdist.tensor_parallel_rank() == 0:
        np.testing.assert_allclose(out, out_tp)
