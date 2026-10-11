r"""Optimizer parameter groups: weight-decay rules (none on biases, other 1-D parameters or named parameters)
and per-keyword overrides of the optimizer options, such as a lower learning rate for one sub-module."""

from typing import Any, Callable, Dict, List, Mapping, Sequence, Union

from torch import Tensor, nn

NoDecayRule = Union[bool, Sequence[str], Callable[[str, Tensor], bool]]

__all__: list = []


def _contains(keyword: str) -> Callable[[str, Tensor], bool]:
    r"""Matcher of the parameters whose name contains `keyword`."""

    def matcher(name: str, param: Tensor) -> bool:
        return keyword in name

    return matcher


def _no_decay_rule(rule: NoDecayRule) -> Callable[[str, Tensor], bool]:
    r"""Resolve the `no_decay` option into a predicate on `(name, param)`."""
    if callable(rule):
        return rule

    keywords = tuple(rule) if not isinstance(rule, bool) else ()

    def predicate(name: str, param: Tensor) -> bool:
        # Biases and other 1-D parameters (norm weights, tokens) are excluded from the decay.
        return param.ndim <= 1 or name.endswith(".bias") or any(keyword in name for keyword in keywords)

    return predicate


def param_groups(
    model: nn.Module,
    *,
    weight_decay: float = 0.0,
    no_decay: NoDecayRule = False,
    overrides: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    r"""Split a model's trainable parameters into optimizer groups with weight-decay rules and keyword overrides.

    Every parameter with `requires_grad` goes to the first override whose `keyword` is a substring of its name
    (or whose `match(name, param)` returns `True`), else to the base group. Each group then carries `weight_decay` and the override's other options (`lr`, `momentum`, ...).
    With `no_decay`, every group is further split into a decaying and a non-decaying part: `True` excludes
    biases and 1-D parameters, a list of keywords extends that rule to the names containing them, and a callable
    decides per `(name, param)`. An override that sets its own
    `weight_decay` is not split.

    The groups come out base first, then the overrides in order, each as (decay, no-decay) when split; empty
    groups are dropped. That order is what a per-group `max_lr` list of `OneCycleLR` refers to.

    Args:
        model: Source model.
        weight_decay: Weight decay of the base group (and of overrides that do not set their own).
        no_decay: Rule selecting the parameters that get no weight decay (see above).
        overrides: Dicts, each with `keyword` (substring of the parameter name) or `match`
            (callable `(name, param) -> bool`) plus the optimizer options of that group.

    Returns:
        Optimizer parameter-group dicts, each with `params`, `weight_decay` and the override options.

    Example:
        ```pycon
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))
        >>> groups = param_groups(model, weight_decay=0.05, no_decay=True)
        >>> [(len(g["params"]), g["weight_decay"]) for g in groups]
        [(2, 0.05), (4, 0.0)]
        >>> groups = param_groups(model, weight_decay=0.05, overrides=[dict(keyword="2.", lr=1e-4)])
        >>> [(len(g["params"]), g.get("lr")) for g in groups]
        [(4, None), (2, 0.0001)]

        ```
    """
    matchers: List[Callable[[str, Tensor], bool]] = []
    for override in overrides:
        if "keyword" in override:
            matchers.append(_contains(str(override["keyword"])))
        elif "match" in override:
            matchers.append(override["match"])
        else:
            raise ValueError("Each override needs a `keyword` (substring of the parameter name) or a `match` callable.")

    buckets: List[List[tuple]] = [[] for _ in range(len(overrides) + 1)]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        index = next((i + 1 for i, matcher in enumerate(matchers) if matcher(name, param)), 0)
        buckets[index].append((name, param))

    predicate = _no_decay_rule(no_decay) if no_decay else None
    groups: List[Dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        options = {k: v for k, v in (overrides[index - 1].items() if index else ()) if k not in ("keyword", "match")}
        options.setdefault("weight_decay", weight_decay)
        split = predicate is not None and (index == 0 or "weight_decay" not in overrides[index - 1])
        decay = [p for n, p in bucket if not (split and predicate(n, p))]  # type: ignore[misc]
        skipped = [p for n, p in bucket if split and predicate(n, p)]  # type: ignore[misc]
        if decay:
            groups.append({"params": decay, **options})
        if skipped:
            groups.append({"params": skipped, **{**options, "weight_decay": 0.0}})
    return groups
