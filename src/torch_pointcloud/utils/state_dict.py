import re
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Set

from torch import Tensor

_PLACEHOLDER_RE = re.compile(r"^\s*(\w+)\s*(?:([+-])\s*(\d+))?\s*$")


def _resolve_placeholder(expr: str, ctx: Dict[str, Any]) -> Any:
    match = _PLACEHOLDER_RE.match(expr)
    if match is None or match.group(1) not in ctx:
        raise ValueError(f"Unsupported placeholder {{{expr}}}: expected a captured name, optionally with `+N`/`-N`.")

    value = ctx[match.group(1)]
    if match.group(2) is None:
        return value

    offset = int(match.group(3))
    return value + offset if match.group(2) == "+" else value - offset


def transform_state_dict(
    state_dict: Dict[str, Any],
    mapping: Dict[str, str],
    value_transform: Optional[Callable[[Tensor], Tensor]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Transform a pytorch module state dict by remapping keys and optionally transforming associated tensors.
    This function is designed to map the state dict of a pytorch module to a different state dict,
    facilitating the transfer of weights between different models.

    Args:
        state_dict: The state dict to transform.
        mapping: A dictionary mapping the old keys to the new keys.
        value_transform: A function to transform the values.

    Returns:
        The transformed state dict.

    Example:
        ```python
        import torch
        from torch_pointcloud.utils.state_dict import transform_state_dict

        state_dict = {
            "encoder.conv.0.weight": torch.randn(1, 3, 16, 16),
            "encoder.conv.0.bias": torch.randn(1),
            "encoder.norm.1.weight": torch.randn(1, 16, 16, 16),
            "encoder.norm.1.bias": torch.randn(1),
            "encoder.norm.1.running_mean": torch.randn(16),
            "encoder.norm.1.running_var": torch.randn(16),
        }
        mapping = {
            "encoder.{module}.{i}.weight": "backbone.{module}.{i+1}.weight",
            "encoder.{module}.{i}.bias": "backbone.{module}.{i+1}.bias",
            "encoder.{module}.{i}.running_{stat}": "backbone.{module}.{i+1}.running_{stat}",
        }
        state_dict = transform_state_dict(state_dict, mapping)
        print(state_dict.keys())
        # {
        #     "backbone.conv.1.weight": ...,
        #     "backbone.conv.1.bias": ...,
        #     "backbone.norm.2.weight": ...,
        #     "backbone.norm.2.bias": ...,
        #     "backbone.norm.2.running_mean": ...,
        #     "backbone.norm.2.running_var": ...,
        # }
        ```
    """
    value_transform = value_transform or (lambda v: v)

    # build the rules for the key transformation / mapping
    rules = []
    for src, dst in mapping.items():
        pattern = re.escape(src).replace(r"\{", "{").replace(r"\}", "}")
        pattern = re.sub(r"\{(\w+):int\}", r"(?P<\1>\\d+)", pattern)
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^.]+)", pattern)
        rules.append((re.compile(f"^{pattern}$"), dst))

    # Track which rules matched so `strict` can report mapping patterns that never applied.
    used_rule_idxs: Set[int] = set()

    def key_transform(key: str) -> str:
        for i, (pattern, template) in enumerate(rules):
            if match := pattern.match(key):
                # track which rule (i.e. mapping pattern) was used
                used_rule_idxs.add(i)

                # cast to int if possible to allow for arithmetic operations
                # e.g. "param.{i}.weights" -> "param.{i+1}.weights"
                ctx = {k: (int(v) if v.isdigit() else v) for k, v in match.groupdict().items()}
                return re.sub(r"\{([^}]+)\}", lambda m: str(_resolve_placeholder(m.group(1), ctx)), template)
        return key

    transformed_state_dict = [(key_transform(k), value_transform(v)) for k, v in state_dict.items()]
    mapping_keys = list(mapping.keys())
    unused_keys = [mapping_keys[i] for i in range(len(rules)) if i not in used_rule_idxs]

    if strict and unused_keys:
        # TODO: Maybe provide a better exception, just like how pytorch does when loading a state dict with unexpected keys.
        # TODO: This way it will be possible to programmatically catch the keys that were not used.
        raise ValueError(
            f"Unused keys found in mapping: {', '.join([f'{k!r}' for k in unused_keys])}.\n"
            "These patterns did not match any keys in the provided state_dict. "
            "You can disable this behavior by setting `strict=False`."
        )

    return OrderedDict(transformed_state_dict)
