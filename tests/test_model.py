import pytest

import numpy as np

from numpitron.nn.core import Layer
from numpitron.nn.linear import Linear
from numpitron.nn.model import Model
from numpitron import distributed as npdist

npdist.init(tp_size=npdist.world_size())


class DummyModel(Model):
    """Dummy model class with single linear layer for testing."""

    def __init__(self, d_in, d_out, **kwargs):
        super().__init__()
        self.add_setting("d_in", d_in)
        self.add_setting("d_out", d_out)
        self.add_layer(
            "linear", Linear(self.d_in, self.d_out, weight_init="zeros", use_bias=False, weight_shard_axis=1)
        )


def test_add_layer():
    model = Model()
    model.add_layer("test_layer", Layer())

    assert hasattr(model, "test_layer")


def test_to_dict():
    model = Model()
    model.add_layer("test_layer", Layer())

    model_dict = model.to_dict()

    assert "settings" in model_dict
    assert "layers" in model_dict
    assert "test_layer" in model_dict["layers"]


def test_from_dict():
    model_dict = {
        "settings": {"d_in": 1, "d_out": 2},
        "layers": {
            "linear": {
                "settings": {"d_in": 1, "d_out": 2, "use_bias": False, "weight_init": "zeros"},
                "parameters": {
                    "weight": {"data": np.zeros((1, 2)), "shard_axis": None}
                },
            }
        },
    }

    model = DummyModel.from_dict(model_dict)

    assert hasattr(model, "linear")
    assert isinstance(model.linear, Linear)


def test_to_from_dict():
    model_dict = DummyModel(4, 5).to_dict()
    model = DummyModel.from_dict(model_dict)

    assert hasattr(model, "linear")
    assert isinstance(model.linear, Linear)
    assert model.linear.weight.data.shape == (4, 5)


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_scatter_all_gather():
    model = DummyModel(6, 6)
    model.linear.weight_shard_axis = 1

    model.scatter()

    assert model.linear.weight.data.shape == (6, 3)
    assert model.is_scattered

    model.all_gather()

    assert model.linear.weight.data.shape == (6, 6)
    assert not model.is_scattered
