import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn.parameter import is_lazy

from .types import PathLike

_PLACEHOLDER_RE = re.compile(r"^\s*(\w+)\s*(?:([+-])\s*(\d+))?\s*$")


def read_state_dict(path: PathLike) -> Dict[str, Any]:
    """Read a checkpoint file into a flat state dict.

    Supports `.safetensors` and `torch.save` files, unwrapping a `state_dict` key if present. Lightning
    checkpoints (identified by their `pytorch-lightning_version` key) are reduced to the wrapped network:
    only `model.`-prefixed keys are kept, with the prefix stripped.

    Args:
        path: Checkpoint file to read.

    Returns:
        The flat parameter-name to tensor mapping.

    Example:
        ```{.python notest}
        from torch_pointcloud.utils.state_dict import read_state_dict

        state_dict = read_state_dict("checkpoints/last.ckpt")
        ```
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        return dict(load_file(path.as_posix()))

    data = torch.load(path, weights_only=True, map_location="cpu")
    state_dict = data["state_dict"] if "state_dict" in data else data
    if "pytorch-lightning_version" in data:
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}

    return state_dict


def load_state_dict(model: nn.Module, state_dict: Dict[str, Any], source: str) -> None:
    """Load `state_dict` into `model`, adapting the head instead of failing on a rebuilt classifier.

    Checkpoint keys absent from the model are ignored with a warning (e.g. head weights when the model was
    built with `num_classes=0`), keys with mismatched tensor shapes are skipped with a warning and keep their
    fresh initialization (e.g. after a `num_classes` override), and model keys missing from the checkpoint
    raise. A checkpoint matching the model exactly loads completely, as with a strict load.

    Args:
        model: Module to load the parameters into.
        state_dict: Flat parameter-name to tensor mapping (see `read_state_dict`).
        source: Label identifying where the checkpoint came from, used in warnings and errors.

    Raises:
        RuntimeError: If the checkpoint is missing keys the model requires.

    Example:
        ```{.python notest}
        from torch_pointcloud.utils.state_dict import load_state_dict, read_state_dict

        load_state_dict(model, read_state_dict("checkpoints/last.ckpt"), source="last.ckpt")
        ```
    """
    model_state = model.state_dict()
    unexpected = sorted(key for key in state_dict if key not in model_state)
    mismatched = sorted(
        key
        for key in state_dict
        if key in model_state and not is_lazy(model_state[key]) and state_dict[key].shape != model_state[key].shape
    )
    missing = sorted(key for key in model_state if key not in state_dict)
    if missing:
        raise RuntimeError(f"Checkpoint {source!r} is missing model keys: {', '.join(missing)}.")
    if unexpected:
        warnings.warn(f"Ignoring checkpoint keys from {source!r} absent from the model: {', '.join(unexpected)}.")
    if mismatched:
        warnings.warn(
            f"Skipping checkpoint keys from {source!r} with mismatched shapes, keeping their initialization: "
            f"{', '.join(mismatched)}."
        )

    dropped = set(unexpected) | set(mismatched)
    model.load_state_dict({key: value for key, value in state_dict.items() if key not in dropped}, strict=False)


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
        strict: Raise a `ValueError` on mapping patterns that match no key and on source keys colliding
            onto the same destination key. When `False`, unused patterns are ignored and collisions only
            emit a warning.

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

    first_source: Dict[str, str] = {}
    collisions = []
    for src_key, (dst_key, _) in zip(state_dict, transformed_state_dict):
        if dst_key in first_source:
            collisions.append(f"{first_source[dst_key]!r} and {src_key!r} -> {dst_key!r}")
        else:
            first_source[dst_key] = src_key

    if collisions:
        message = (
            f"Colliding keys found in mapping: {'; '.join(collisions)}.\n"
            "Multiple source keys map to the same destination key, so all but the last value are lost."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message)

    return OrderedDict(transformed_state_dict)
