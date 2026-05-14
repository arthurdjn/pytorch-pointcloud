import torch.nn as nn

from torch_pointcloud.layers.layer_container import LayerContainer


class _Encoder(LayerContainer):
    layer_name = "layer"


def test_layer_container_add_and_get() -> None:
    enc = _Encoder()
    assert enc.num_layers == 0

    l0 = nn.Linear(4, 8)
    l1 = nn.Linear(8, 16)
    enc.add_layer(l0)
    enc.add_layer(l1)

    assert enc.num_layers == 2
    assert enc.get_layer(0) is l0
    assert enc.get_layer(1) is l1


def test_layer_container_iter_layers() -> None:
    enc = _Encoder()
    layers = [nn.Linear(4, 8), nn.Linear(8, 16), nn.Linear(16, 32)]
    for layer in layers:
        enc.add_layer(layer)

    iterated = list(enc.iter_layers())
    assert iterated == layers
