from functools import partial, wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

import torch.nn as nn

ModelFactory = Callable[..., nn.Module]
ModelConfig = Dict[str, Any]
T = TypeVar("T", bound=ModelFactory)

_model_registry: Dict[str, ModelFactory] = {}


@overload
def register_model(fn: T) -> T: ...


@overload
def register_model(fn: None = None, cfg: Optional[ModelConfig] = None) -> Callable[[T], T]: ...


def register_model(
    fn: Optional[ModelFactory] = None, cfg: Optional[ModelConfig] = None
) -> Union[ModelFactory, Callable[[ModelFactory], ModelFactory]]:
    """Register a model variant function.

    This decorator can be used in two ways:
        1. Plain decorator: @register_model
        2. Decorator with config: @register_model(cfg=...)

    Args:
        fn: The model creation function to register
        cfg: Optional configuration for the model variant

    Returns:
        The decorated function or a partial decorator if used with parameters.

    Examples:
        You can use this decorator in two ways:

        1. Plain decorator:

            ```python
            @register_model
            def model() -> nn.Module:
                ...
            ```

        2. Decorator with config:

            ```python
            @register_model(cfg={'key': 'value'})
            def model() -> nn.Module:
                ...
            ```
    """
    if fn is None:
        return partial(register_model, cfg=cfg)

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> nn.Module:
        return fn(*args, **kwargs)

    model_name = fn.__name__
    _model_registry[model_name] = wrapper

    return wrapper
