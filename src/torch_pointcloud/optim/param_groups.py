r"""Optimizer parameter groups: weight-decay rules (none on biases, other 1-D parameters or named parameters)
and per-keyword overrides of the optimizer options, such as a lower learning rate for one sub-module."""

from typing import Any, Callable, Dict, List, Literal, Mapping, Sequence, Union

from torch import Tensor, nn

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size

MatchType = Literal["select", "filter"]
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


def generate_param_groups(
    network: nn.Module,
    layer_matches: Sequence[Callable[..., Any]],
    match_types: Union[MatchType, Sequence[MatchType]],
    lr_values: Union[float, Sequence[float]],
    include_others: bool = True,
) -> List[Dict[str, Any]]:
    r"""Split a network's parameters into optimizer parameter groups with per-group LRs.

    Mirrors :github: [`monai.optimizers.generate_param_groups`](https://docs.monai.io/en/stable/optimizers.html#generate-param-groups):
    for each $(\text{layer\_matches}[i], \text{match\_types}[i], \text{lr\_values}[i])$:

    - `"select"`: `layer_matches[i](network)` returns a `Module`; its parameters
      form group $i$.
    - `"filter"`: `layer_matches[i](name)` is called for every parameter name and
      returns a `bool`; matching parameters join group $i$ (`fnmatch.fnmatchcase` and friends drop in).

    Scalar `match_types` / `lr_values` are broadcast to the length of `layer_matches`.
    Groups may overlap (a parameter can appear in multiple groups). When
    `include_others=True`, a final group collects parameters not matched by *any*
    group, with no LR override.

    Args:
        network: Source network.
        layer_matches: Matcher callables, one per group.
        match_types: `"select"` or `"filter"` (per group, or a single value broadcast).
        lr_values: Per-group LR (per group, or a single value broadcast).
        include_others: Append a final group with the unmatched parameters.

    Returns:
        Optimizer parameter-group dicts: matched groups in order, then `"others"`
        (when included).
    """
    layer_matches = ensure_tuple(layer_matches)
    match_types = ensure_tuple_size(match_types, size=len(layer_matches))
    lr_values = ensure_tuple_size(lr_values, size=len(layer_matches))

    def _get_select(f: Callable[[nn.Module], nn.Module]) -> Callable[[], Any]:
        def _select() -> Any:
            return f(network).parameters()

        return _select

    def _get_filter(f: Callable[[str], bool]) -> Callable[[], Any]:
        def _filter() -> Any:
            return (p for n, p in network.named_parameters() if f(n))

        return _filter

    params: List[Dict[str, Any]] = []
    matched_ids: List[int] = []
    for func, ty, lr in zip(layer_matches, match_types, lr_values):
        kind = ty.lower()
        if kind == "select":
            layer_params = _get_select(func)
        elif kind == "filter":
            layer_params = _get_filter(func)
        else:
            raise ValueError(f"Unsupported layer match type: {ty!r}; expected 'select' or 'filter'.")

        params.append({"params": list(layer_params()), "lr": lr})
        matched_ids.extend(id(p) for p in layer_params())

    if include_others:
        params.append({"params": [p for p in network.parameters() if id(p) not in matched_ids]})

    return params
