"""Activation factory wrapper around :pyg:`torch_geometric.nn.resolver`.

Thin wrapper around `torch_geometric.nn.resolver.activation_resolver` that returns
`None` when `act` is `None`, so callers don't need a `None` guard before assigning
the result. Exists for API symmetry with `create_norm`, which adds dimensionality
support that PyG's resolver does not have.
"""

from typing import Any, Callable, Optional, Union

import torch.nn as nn
from torch_geometric.nn.resolver import activation_resolver


def create_act(act: Union[str, Callable, None], **act_kwargs: Any) -> Optional[nn.Module]:
    """Resolve an activation, returning `None` for `act=None`.

    Args:
        act: Activation name (e.g., `"relu"`, `"leaky_relu"`), a class, an instance,
            or `None`.
        **act_kwargs: Forwarded to the activation constructor (ignored if `act` is
            already an instance).

    Returns:
        The instantiated activation module, or `None` if `act is None`.
    """
    if act is None:
        return None
    return activation_resolver(act, **act_kwargs)
