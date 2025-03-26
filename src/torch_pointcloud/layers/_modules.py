from functools import partial
from typing import Any, Callable, Dict, Type, TypeVar, Union

import torch.nn as nn

ModuleName = TypeVar("ModuleName", bound=str)

# NOTE: MyPy is strict about partial types, so we need to use `Any` here.
ModuleLike = Union[Type[nn.Module], nn.Module, Callable, partial[Any], ModuleName]
RegisteredModuleLike = Union[Type[nn.Module], partial[Any]]

# NOTE: We want to allow specifying registry dict with literal string keys,
# to provide better type hints support.
ModuleRegistryDict = Dict[ModuleName, RegisteredModuleLike]


def create_module(name: ModuleLike, *args: Any, registry: ModuleRegistryDict, **kwargs: Any) -> nn.Module:
    """Utility function to instantiate modules from a registry, or directly pass a module instance.

    Args:
        name: The name of the module to instantiate, or the module type or instance itself.
        args: Positional arguments to pass to the module constructor.
        registry: A dictionary mapping module names to their types.
        kwargs: Keyword arguments to pass to the module constructor.

    Returns:
        The instantiated module.
    """
    if isinstance(name, nn.Module):
        # Skip registry lookup, in case custom modules are passed directly
        return name
    elif isinstance(name, str):
        # Lookup module in registry
        module = registry.get(name)
        if module is None:
            available_names = ", ".join(registry.keys())
            raise ValueError(f"Could not find module with name {name!r}. Available modules: {available_names}")
        return module(*args, **kwargs)
    # Instantiate (partial) modules
    return name(*args, **kwargs)
