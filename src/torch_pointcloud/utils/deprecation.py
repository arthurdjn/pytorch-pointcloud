"""Helpers that keep renamed public names working, with a `FutureWarning`, for a deprecation period."""

import warnings
from typing import Mapping


def renamed_attribute(name: str, renamed: Mapping[str, str], module: str) -> str:
    """Return the current name of `name`, a former attribute of `module`, with a `FutureWarning`.

    Call it from the module-level `__getattr__` of the module that renamed the attribute, which Python only reaches
    for a name the module does not define. The warning points at the code that read the former name.

    Args:
        name: Attribute name being read.
        renamed: Former names mapped to their current names in `module`.
        module: Name of the module, `__name__`, quoted in the `AttributeError`.

    Returns:
        The current name of `name`.

    Raises:
        AttributeError: If `name` is not a former name in `renamed`.

    Example:
        At the end of a module that renamed `old_function` to `new_function`:

        ```python
        from typing import Any

        from torch_pointcloud.utils.deprecation import renamed_attribute

        _RENAMED = {"old_function": "new_function"}


        def __getattr__(name: str) -> Any:
            return globals()[renamed_attribute(name, _RENAMED, __name__)]
        ```
    """
    if name not in renamed:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    # Frame 3 is the code that reads the former name: this function, then the module `__getattr__`, then it.
    warnings.warn(f"`{name}` is deprecated; use `{renamed[name]}` instead.", FutureWarning, stacklevel=3)
    return renamed[name]
