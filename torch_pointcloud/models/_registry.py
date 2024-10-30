from functools import partial, wraps
from typing import Any, Callable, Dict, Optional, Type, Union

import torch.nn as nn

_model_registry: Dict[str, Any] = {}


def register_model(fn=None, default_cfg=None):
    """Register a model variant function.
    
    Args:
        fn: The model creation function to register
        default_cfg: Default configuration for the model variant
    """
    if fn is None:
        # Called with parameters: @register_model(default_cfg=...)
        return partial(register_model, default_cfg=default_cfg)
    
    # Called directly: @register_model
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    
    # Register the model
    model_name = fn.__name__
    _model_registry[model_name] = fn
    if default_cfg is not None:
        _model_configs[model_name] = default_cfg
    
    return wrapper