from functools import partial
from typing import Any, Dict, Type, Union

import torch.nn as nn

ACT_LAYERS: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "none": nn.Identity,
}


def get_act_layer(name: Union[Type[nn.Module], str], **kwargs: Any) -> nn.Module:
    if isinstance(name, str):
        return ACT_LAYERS[name](**kwargs)
    return name(**kwargs)
