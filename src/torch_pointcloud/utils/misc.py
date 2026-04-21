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
    """Apply ``func`` to every item in ``iterable``, optionally in parallel.

    When ``num_workers`` is ``None``, items are processed sequentially in the
    current process. Otherwise ``joblib`` spawns ``num_workers`` workers and
    results are streamed back as each task completes.

    Args:
        func: Callable invoked on each item.
        iterable: Items to map over.
        num_workers: Number of worker processes. ``None`` disables parallelism.
        total: Item count for the progress bar. Inferred from ``len(iterable)``
            when not provided.
        desc: Progress bar description.
        show_progress: Whether to display a ``tqdm`` progress bar. The bar
            advances on task *completion* (not dispatch), so it stays accurate
            under parallel execution.
    """
    if show_progress and total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    if num_workers is None:
        results: Iterable[T] = (func(item) for item in iterable)
    else:
        results = Parallel(n_jobs=num_workers, return_as="generator")(
            delayed(func)(item) for item in iterable
        )

    if show_progress:
        results = tqdm(results, total=total, desc=desc)

    return list(results)
