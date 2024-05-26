from typing import Self

from numpitron.nn.core import Layer
import numpitron.distributed as npdist


class Model(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.layers = {}

    def add_layer(self, name, layer: Layer) -> None:
        """Add a layer to the model."""
        setattr(self, name, layer)
        self.layers[name] = layer

    def get_layers(self):
        layers = {
            layer_name: layer.get_layers() if isinstance(layer, Model) else layer
            for layer_name, layer in self.layers.items()
        }
        return layers

    def scatter(self, src: int = 0) -> None:
        assert not self.is_scattered, "Cannot scatter an already scattered model."
        for layer in self.layers.values():
            layer.scatter(src)
        self.is_scattered = True

    def all_gather(self) -> None:
        for layer in self.layers.values():
            layer.all_gather()
        self.is_scattered = False

    def to_dict(self) -> dict:
        """Return a dictionary representation of this model."""
        layers = {name: layer.to_dict() for name, layer in self.layers.items()}
        return {"settings": self.settings, "layers": layers}

    @classmethod
    def from_dict(cls, model_dict: dict[str, dict]) -> Self:
        settings, layers = model_dict["settings"], model_dict["layers"]
        settings |= {"weight_init": "zeros", "bias_init": "zeros"}
        model = cls(**settings)

        for name in model.layers:
            assert name in layers, f"Expected {name} in {layers}."

            model.add_layer(name, getattr(model, name).from_dict(layers[name]))
            
            # hacky but parallel context has to be set up already when here.
            if name == 'mlp':
                model.mlp.row_linear.use_bias = npdist.tensor_parallel_rank() == 0
        return model
