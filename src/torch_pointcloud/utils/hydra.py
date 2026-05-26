"""Hydra config helpers.

Thin wrappers around `hydra.utils.instantiate` that handle the dict/list
group-config shapes Hydra produces. `hydra` is a dev/optional dependency, so
the import is deferred inside each function.
"""

from typing import Any, List


def instantiate_list(cfg: Any) -> List[Any]:
    """Instantiate every entry of a Hydra config group into a flat list.

    Accepts:

    - `None` (or any falsy value): returns `[]`.
    - A mapping / `DictConfig` (e.g. `callbacks: {ckpt: {...}, lr: {...}}`):
      instantiates each value, skipping `None` entries.
    - A sequence / `ListConfig` (e.g. `[{_target_: ...}, {_target_: ...}]`):
      instantiates each element, skipping `None` entries.

    Anything else raises `TypeError`.
    """
    from hydra.utils import instantiate

    if not cfg:
        return []
    if hasattr(cfg, "values") and callable(cfg.values):
        nodes = cfg.values()
    elif hasattr(cfg, "__iter__"):
        nodes = cfg
    else:
        raise TypeError(f"Expected a mapping, sequence, or None for instantiate_list, got {type(cfg).__name__}.")
    return [instantiate(node) for node in nodes if node is not None]
