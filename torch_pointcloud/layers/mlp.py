from typing import List, Optional

from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Identity, LeakyReLU, Linear, Module, ReLU, Sequential


def create_mlp(dims: List[int], bn: bool = True, activation: Optional[Module] = None) -> Module:
    if len(dims) < 2:
        raise ValueError(f"The MLP must have at least 2 dimensions. Got {len(dims)}.")

    if activation is None:
        activation = ReLU()

    activation = activation or ReLU()
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layer = Sequential(
            Linear(in_dim, out_dim),
            BatchNorm1d(out_dim) if bn else Identity(),
            activation,
        )
        layers.append(layer)
    return Sequential(*layers)


def create_shared_mlp(
    channels: List[int],
    bias: bool = False,
    bn: bool = True,
    activation: Optional[Module] = None,
) -> Module:
    if len(channels) < 2:
        raise ValueError(f"The SharedMLP2d must have at least 2 channels. Got {len(channels)}.")

    activation = activation or LeakyReLU(negative_slope=0.01)
    layers = []
    for in_channels, out_channels in zip(channels[:-1], channels[1:]):
        layer = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=bias),
            BatchNorm2d(out_channels) if bn else Identity(),
            activation,
        )
        layers.append(layer)
    return Sequential(*layers)


class MLP(Module):
    def __init__(self, dims: List[int], bn: bool = True, activation: Optional[Module] = None) -> None:
        super().__init__()
        self.layers = create_mlp(dims, bn=bn, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


# TODO: possibility to choose the convolutional layer and batch normalization layer
# TODO: Then, create two other utility modules: SharedMLP1d and SharedMLP2d
class SharedMLP(Module):
    """
    Shared Multi-Layer Perceptron (MLP) for dense point clouds.

    Args:
        channels: List of integers. The length of the list is the number of layers in the MLP.
            Each integer is the number of channels in the corresponding layer.
        bias: If set to `True`, the layers will have a bias term.
        bn: If set to `True`, the layers will have a batch normalization.
        activation: The activation function to use.
            If `None`, a leaky ReLU with a negative slope of 0.01 is used.

    Input:
        - x: :math:`(B, C_{in}, M, K)` tensor

    Output:
        - x: :math:`(B, C_{out}, M, K)` tensor

    """

    def __init__(self, channels: List[int], bias: bool = False, bn: bool = True, activation: Optional[Module] = None):
        super().__init__()
        self.layers = create_shared_mlp(channels, bias=bias, bn=bn, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
