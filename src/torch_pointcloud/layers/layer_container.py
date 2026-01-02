from typing import (
    ClassVar,
    Iterator,
)

import torch.nn as nn


class LayerContainer(nn.Module):
    r"""A helper class used to manage a list of layers with a common prefix.
    This class is used for encoder, where all layers are registered with a common prefix and can be accessed by index.

    In opposition to the `torch.nn.ModuleList` where registered layers are indexed by their position in the list,
    this class allows to access layers by their name (e.g. "layer0", "layer1", etc.).
    """

    layer_name: ClassVar[str]

    @property
    def num_layers(self) -> int:
        layer_names = [name for name in self._modules.keys() if name.startswith(self.layer_name)]
        return len(layer_names)

    def add_layer(self, layer: nn.Module) -> None:
        layer_name = f"{self.layer_name}{self.num_layers}"
        self.add_module(layer_name, layer)

    def get_layer(self, index: int) -> nn.Module:
        layer_name = f"{self.layer_name}{index}"
        return self.get_submodule(layer_name)

    def iter_layers(self) -> Iterator[nn.Module]:
        for i in range(self.num_layers):
            yield self.get_layer(i)
