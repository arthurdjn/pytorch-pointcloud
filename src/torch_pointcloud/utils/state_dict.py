"""Checkpoint reading, adaptive `state_dict` loading, and key-mapping transforms."""

import re
import warnings
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn.parameter import is_lazy

from .types import PathLike

_PLACEHOLDER_RE = re.compile(r"^\s*(\w+)\s*(?:([+-])\s*(\d+))?\s*$")


class _PickledObject:
    """Stand-in for a pickled object that is not a tensor or a container, keeping its arguments and state without
    importing or calling the class it was saved from. Each global is read through a subclass named after it."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "_PickledObject":
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        self.state = state


def read_state_dict(path: PathLike) -> Dict[str, Any]:
    """Read a checkpoint file into a flat state dict.

    Supports `.safetensors` and `torch.save` files, unwrapping an `ema_state_dict`, `state_dict` or
    `model_state_dict` key if present, in that order: a training checkpoint that stores an exponential moving average
    of the weights yields the averaged weights. Lightning checkpoints (identified by their `pytorch-lightning_version`
    key) are reduced to the wrapped network: only `model.`-prefixed keys are kept, with the prefix stripped.
    `torch.save` files load with `weights_only=True`; the objects of a zip checkpoint that pickle their class and state
    (training hyperparameters, optimizer and scheduler configurations) are read as inert placeholders, so their classes
    are neither imported nor called.

    Args:
        path: Checkpoint file to read.

    Returns:
        The flat parameter-name to tensor mapping.

    Raises:
        ValueError: If the checkpoint, or its `state_dict` entry, is not a mapping, if it holds such an object among
            the parameters (e.g. a whole module), or if a Lightning checkpoint has no `model.`-prefixed key to
            extract.

    Example:
        ```{.python notest}
        from torch_pointcloud.utils.state_dict import read_state_dict

        state_dict = read_state_dict("checkpoints/last.ckpt")
        ```
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        return dict(load_file(path.as_posix()))

    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(path) if zipfile.is_zipfile(path) else []
    with torch.serialization.safe_globals([(type(name, (_PickledObject,), {}), name) for name in unsafe_globals]):
        data = torch.load(path, weights_only=True, map_location="cpu")

    state_dict = (
        next(
            (data[key] for key in ("ema_state_dict", "state_dict", "model_state_dict") if data.get(key) is not None),
            data,
        )
        if isinstance(data, dict)
        else data
    )
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint {path.as_posix()!r} is not a state dict: expected a mapping of parameter names to "
            f"tensors, got {type(state_dict).__name__}."
        )

    if "pytorch-lightning_version" in data:
        extracted = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
        if state_dict and not extracted:
            sample = ", ".join(list(state_dict)[:5])
            raise ValueError(
                f"Lightning checkpoint {path.as_posix()!r} has no 'model.'-prefixed keys (found: {sample}); "
                "the wrapped network must be stored under `self.model` to be extracted."
            )

        state_dict = extracted

    pickled = [
        f"{key!r} ({type(value).__name__})" for key, value in state_dict.items() if isinstance(value, _PickledObject)
    ]
    if pickled:
        raise ValueError(
            f"Checkpoint {path.as_posix()!r} holds objects other than tensors among its parameters: "
            f"{', '.join(pickled[:5])}; save the `state_dict()` of the module instead."
        )

    return state_dict


def load_state_dict(model: nn.Module, state_dict: Dict[str, Any], source: str, strict: bool = True) -> None:
    """Load `state_dict` into `model`, keeping the initialization of the keys whose shapes do not match.

    Checkpoint keys absent from the model are ignored with a warning, keys with mismatched tensor shapes are
    skipped and keep their fresh initialization (e.g. the stem after an `in_channels` override), and model keys
    missing from the checkpoint raise. A checkpoint matching the model exactly loads completely, as with a
    strict load.

    Args:
        model: Module to load the parameters into.
        state_dict: Flat parameter-name to tensor mapping (see `read_state_dict`).
        source: Label identifying where the checkpoint came from, used in warnings and errors.
        strict: Raise when the checkpoint is missing model keys. Pass `False` to warm-start from a partial
            checkpoint (e.g. a backbone-only or pretraining export): missing keys warn and keep their fresh
            initialization.

    Raises:
        RuntimeError: If `strict` and the checkpoint is missing keys the model requires.

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
        if strict:
            raise RuntimeError(f"Checkpoint {source!r} is missing model keys: {', '.join(missing)}.")

        warnings.warn(
            f"Checkpoint {source!r} is missing model keys, keeping their initialization: {', '.join(missing)}.",
            stacklevel=2,
        )
    if unexpected:
        warnings.warn(
            f"Ignoring checkpoint keys from {source!r} absent from the model: {', '.join(unexpected)}.",
            stacklevel=2,
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
        raise ValueError(
            f"Unused mapping patterns: {', '.join(f'{k!r}' for k in unused_keys)}. "
            "These patterns matched no key in the state dict; pass `strict=False` to ignore them."
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

        warnings.warn(message, stacklevel=2)

    return OrderedDict(transformed_state_dict)
