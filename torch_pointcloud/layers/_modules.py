from functools import partial
from typing import Any, Dict, Type, Union

import torch.nn as nn

MODULE_TYPE = Union[Type[nn.Module], partial[nn.Module], nn.Module, str]
REGISTERED_MODULE_TYPE = Union[Type[nn.Module], partial[nn.Module]]


def get_module(name: MODULE_TYPE, *args: Any, registry: Dict[str, REGISTERED_MODULE_TYPE], **kwargs: Any) -> nn.Module:
    """Utility function to instantiate modules from a registry, or directly pass a module instance.

    Args:
        name: The name of the module to instantiate, or the module type or instance itself.
        args: Positional arguments to pass to the module constructor.
        registry: A dictionary mapping module names to their types.
        kwargs: Keyword arguments to pass to the module constructor.

    Returns:
        The instantiated module.

    Example:
        >>> import torch.nn as nn
        >>> from torch_pointcloud.layers._modules import get_module
        >>> get_module(nn.Conv1d, 3, 4, kernel_size=1, stride=1)
        Conv1d(3, 4, kernel_size=(1,), stride=(1,))

        >>> from functools import partial
        >>> module = partial(nn.Conv1d, kernel_size=1, stride=1)
        >>> get_module(module, 3, 4)
        Conv1d(3, 4, kernel_size=(1,), stride=(1,))

        >>> registry = {"conv1d": nn.Conv1d, "conv2d": nn.Conv2d}
        >>> get_module("conv1d", 3, 4, kernel_size=1, stride=1, registry=registry)
        Conv1d(3, 4, kernel_size=(1,), stride=(1,))
    """
    if isinstance(name, nn.Module):
        return name
    elif isinstance(name, str):
        module = registry.get(name)
        if module is None:
            available_names = ", ".join(registry.keys())
            raise ValueError(f"Could not find module with name '{name}'. Available modules: {available_names}")
        return module(*args, **kwargs)
    # Instantiate (partial) modules
    return name(*args, **kwargs)
