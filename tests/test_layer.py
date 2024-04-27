import pytest

import numpy as np

from numpitron.nn.layer import Layer
from numpitron import distributed as npdist

npdist.init(tp_size=npdist.world_size())


def test_add_parameter():
    data = np.ones((1,))
    layer = Layer()
    layer.add_parameter("test_weight", data)

    np.testing.assert_array_equal(layer.test_weight.data, data)
    np.testing.assert_array_equal(layer.parameters["test_weight"].data, data)


def test_update_parameter():
    data = np.zeros((1,))
    layer = Layer()
    layer.add_parameter("test_weight", data)
    layer.update_parameter("test_weight", data=data + 1)

    np.testing.assert_array_equal(layer.test_weight.data, data + 1)
    np.testing.assert_array_equal(layer.parameters["test_weight"].data, data + 1)


def test_to_dict():
    layer = Layer()
    layer.add_parameter("test_weight1", np.ones((5,)))
    layer.add_parameter("test_weight2", np.ones((20, 20)), 1)
    layer_dict = layer.to_dict()
    layer_dict = layer_dict["parameters"]

    np.testing.assert_array_equal(layer_dict["test_weight1"]["data"], np.ones((5,)))
    np.testing.assert_array_equal(layer_dict["test_weight2"]["data"], np.ones((20, 20)))
    assert layer_dict["test_weight1"]["gradient"] is None
    assert layer_dict["test_weight2"]["gradient"] is None
    assert layer_dict["test_weight1"]["shard_axis"] is None
    assert layer_dict["test_weight2"]["shard_axis"] == 1


def test_from_dict():
    layer = Layer.from_dict(
        {
            "settings": {},
            "parameters": {
                "test_weight1": {
                    "data": np.ones((5,)),
                    "gradient": None,
                    "shard_axis": None,
                },
                "test_weight2": {
                    "data": np.ones((20, 20)),
                    "gradient": None,
                    "shard_axis": 1,
                },
            },
        }
    )

    np.testing.assert_array_equal(layer.test_weight1.data, np.ones((5,)))
    np.testing.assert_array_equal(layer.test_weight2.data, np.ones((20, 20)))
    assert layer.test_weight1.gradient is None
    assert layer.test_weight2.gradient is None
    assert layer.test_weight1.shard_axis is None
    assert layer.test_weight2.shard_axis == 1


def test_to_from_dict():
    layer = Layer()
    layer.add_parameter("test_weight1", np.ones((5,)))
    layer.add_parameter("test_weight2", np.ones((20, 20)))
    layer_dict = layer.to_dict()
    new_layer = Layer.from_dict(layer_dict)

    np.testing.assert_array_equal(layer.test_weight1.data, new_layer.test_weight1.data)
    np.testing.assert_array_equal(layer.test_weight2.data, new_layer.test_weight2.data)


@pytest.mark.parametrize("shard_axis", [0, 1])
@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_scatter(shard_axis: int):
    layer = Layer()
    layer.add_parameter("test_weight", np.ones((20, 30)), shard_axis=shard_axis)
    layer.update_parameter("test_weight", gradient=np.ones_like(layer.test_weight.data))

    layer.scatter()

    expected_shape = (10, 30) if shard_axis == 0 else (20, 15)
    assert layer.test_weight.data.shape == expected_shape
    assert layer.test_weight.gradient.shape == expected_shape
    assert layer.is_scattered


@pytest.mark.parametrize("shard_axis", [0, 1])
@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_all_gather(shard_axis):
    layer = Layer()
    input_shape = (10, 30) if shard_axis == 0 else (20, 15)
    layer.add_parameter("test_weight", np.ones(input_shape), shard_axis=shard_axis)
    layer.update_parameter("test_weight", gradient=np.ones_like(layer.test_weight.data))
    layer.is_scattered = True

    layer.all_gather()

    assert layer.test_weight.data.shape == (20, 30)
    assert layer.test_weight.gradient.shape == (20, 30)
    assert not layer.is_scattered


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_scatter_without_shard_axis():
    layer = Layer()
    layer.add_parameter("test_weight", np.ones((20, 20)), shard_axis=None)
    layer.scatter()

    assert layer.test_weight.data.shape == (20, 20)


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_gather_without_shard_axis():
    layer = Layer()
    layer.add_parameter("test_weight", np.ones((20, 20)), shard_axis=None)
    layer.is_scattered = True
    layer.all_gather()

    assert layer.test_weight.data.shape == (20, 20)
