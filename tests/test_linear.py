import numpy as np

from numpitron.nn import Linear


def test_linear():
    b, s, d = 32, 64, 128
    
    inputs = np.ones((b, s, d))
    layer = Linear(d, d, init_fn=lambda shape: np.ones(shape))
    outputs = layer(inputs)

    np.testing.assert_allclose(outputs.sum(), b * s * d * d + b * s * d)


def test_linear_no_bias():
    b, s, d = 32, 64, 128
    
    inputs = np.ones((b, s, d))
    layer = Linear(d, d, use_bias=False, init_fn=lambda shape: np.ones(shape))
    outputs = layer(inputs)

    np.testing.assert_allclose(outputs.sum(), b * s * d * d)


def test_row_linear():
    pass


def test_column_linear():
    pass
