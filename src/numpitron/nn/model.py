from typing import Self

from numpitron.nn.layer import Layer


class Model(Layer):
    """Simple wrapper around a block of layers, mainly to ensure
    that scatter and all_gather works as intended."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = {}

    def add_layer(self, name, layer: Layer) -> None:
        """Add a layer to the model."""
        setattr(self, name, layer)
        self.layers[name] = layer

    def scatter(self, src: int = 0) -> None:
        for layer in self.layers:
            layer.scatter(src)

    def all_gather(self) -> None:
        for layer in self.layers:
            layer.all_gather()

    def to_dict(self) -> dict:
        """Return a dictionary representation of this model."""
        layers = {name: layer.to_dict() for name, layer in self.layers.items()}
        return {"settings": self.settings, "layers": layers}

    @classmethod
    def from_dict(cls, model_dict: dict[str, dict]) -> Self:
        settings, layers = model_dict["settings"], model_dict["layers"]
        model = cls(**settings)
        for name, layer in layers.items():
            model.add_layer(name, getattr(model, name).from_dict(layer))
        return model
