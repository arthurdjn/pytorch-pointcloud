"""Miscellaneous helpers for parallel mapping and nested attribute access."""

from collections.abc import Mapping
from typing import Any, Callable, Iterable, List, Optional, TypeVar

from joblib import Parallel, delayed
from tqdm import tqdm

T = TypeVar("T")


def parallel_map(
    func: Callable[..., T],
    iterable: Iterable[Any],
    num_workers: Optional[int] = None,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    show_progress: bool = False,
) -> List[T]:
    """Apply `func` to every item in `iterable`, optionally in parallel.

    When `num_workers` is `None` or `0`, items are processed sequentially in the
    current process. Otherwise `joblib` spawns `num_workers` workers and
    results are streamed back as each task completes.

    Args:
        func: Callable invoked on each item.
        iterable: Items to map over.
        num_workers: Number of worker processes. `None` or `0` disables parallelism.
        total: Item count for the progress bar. Inferred from `len(iterable)`
            when not provided.
        desc: Progress bar description.
        show_progress: Whether to display a `tqdm` progress bar. The bar
            advances on task *completion* (not dispatch), so it stays accurate
            under parallel execution.
    """
    if show_progress and total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    if not num_workers:
        results: Iterable[T] = (func(item) for item in iterable)
    else:
        results = Parallel(n_jobs=num_workers, return_as="generator")(delayed(func)(item) for item in iterable)

    if show_progress:
        results = tqdm(results, total=total, desc=desc)

    return list(results)


def deep_getattr(obj: Any, path: str, default: Any = None) -> Any:
    """Resolve a dotted `path` through nested `Mapping` keys and object attributes, else `default`.

    At each step a `Mapping` (e.g. a batch dict) is indexed by key and any other object is read by
    attribute, so one call resolves both `deep_getattr(batch, "octree.depth")` (dict then attribute) and
    `deep_getattr(module, "criterion.ignore_index")` (attribute then attribute). A missing key or
    attribute (or a `None` reached mid-path) returns `default` rather than raising.

    Args:
        obj: Root object: a `Mapping`, an arbitrary object, or any nesting of the two.
        path: Dotted access path, e.g. `"criterion.ignore_index"`.
        default: Value returned when a segment cannot be resolved.

    Returns:
        The resolved value, or `default` if the path is not fully traversable.
    """
    head, _, rest = path.partition(".")
    if isinstance(obj, Mapping):
        if head not in obj:
            return default
        value = obj[head]
    elif hasattr(obj, head):
        value = getattr(obj, head)
    else:
        return default
    return deep_getattr(value, rest, default) if rest else value
