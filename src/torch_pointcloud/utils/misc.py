from typing import Any, Callable, Iterable, List, Optional, TypeVar

from joblib import Parallel, delayed

T = TypeVar("T")


def parallel_map(func: Callable[..., T], iterable: Iterable[Any], num_workers: Optional[int] = None) -> List[T]:
    if num_workers is None:
        return [func(item) for item in iterable]

    with Parallel(n_jobs=num_workers) as parallel:
        return parallel(delayed(func)(item) for item in iterable)
