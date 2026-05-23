"""Optimizer utilities: parameter-group construction.

`generate_param_groups` mirrors :github: [`monai.optimizers.generate_param_groups`](https://docs.monai.io/en/stable/optimizers.html#generate-param-groups):
parallel sequences of matcher callables, match strategies, and per-group LR values,
with a trailing "others" group for unmatched parameters. Scalar `match_types` or
`lr_values` are broadcast to the length of `layer_matches`.

`"filter"` matchers take the parameter *name* (a `str`) and return a `bool`, so
`fnmatch.fnmatchcase` and friends drop in directly via Hydra's `_partial_`:

```yaml
layer_matches:
  - _target_: fnmatch.fnmatchcase
    _partial_: true
    pat: "*block*"
```
"""

from typing import Any, Callable, Dict, List, Literal, Sequence, Union

from torch import nn

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size

MatchType = Literal["select", "filter"]


def generate_param_groups(
    network: nn.Module,
    layer_matches: Sequence[Callable[..., Any]],
    match_types: Union[MatchType, Sequence[MatchType]],
    lr_values: Union[float, Sequence[float]],
    include_others: bool = True,
) -> List[Dict[str, Any]]:
    r"""Split a network's parameters into optimizer parameter groups with per-group LRs.

    For each $(\text{layer\_matches}[i], \text{match\_types}[i], \text{lr\_values}[i])$:

    - `"select"`: `layer_matches[i](network)` returns a `Module`; its parameters
      form group $i$.
    - `"filter"`: `layer_matches[i](name)` is called for every parameter name and
      returns a `bool`; matching parameters join group $i$.

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
