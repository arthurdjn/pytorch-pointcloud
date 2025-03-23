from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

from joblib import Parallel, delayed

T = TypeVar("T")


def parallel_map(func: Callable[..., T], iterable: Iterable[Any], num_workers: Optional[int] = None) -> List[T]:
    if num_workers is None:
        return [func(item) for item in iterable]

    with Parallel(n_jobs=num_workers) as parallel:
        return parallel(delayed(func)(item) for item in iterable)


def remap_classes(class_to_idx: Dict[str, int], keep_negative: bool = False) -> Dict[str, int]:
    """Remaps class indices to a continuous range starting from 0.

    Args:
        class_to_idx: Dictionary mapping class names to their indices
        keep_negative: If True, negative values remain unchanged

    Returns:
        Dictionary with remapped continuous indices

    Examples:
        >>> remap_classes({"apple": 1, "banana": 5, "orange": 3, "unknown": -1})
        {"apple": 1, "banana": 3, "orange": 2, "unknown": 0}
        >>> remap_classes({"apple": 1, "banana": 5, "orange": 3, "unknown": -1}, keep_negative=True)
        {"apple": 0, "banana": 2, "orange": 1, "unknown": -1}
    """
    # Get unique values and sort them
    values = sorted(list(set(class_to_idx.values())))

    if keep_negative:
        # Split positive and negative values
        neg_values = [v for v in values if v < 0]
        pos_values = [v for v in values if v >= 0]

        # Create mapping for positive values only
        val_to_new = {v: i for i, v in enumerate(pos_values)}
        # Keep negative values as is
        val_to_new.update({v: v for v in neg_values})
    else:
        # Simple consecutive mapping for all values
        val_to_new = {v: i for i, v in enumerate(values)}

    # Apply mapping while preserving keys order
    return {k: val_to_new[v] for k, v in class_to_idx.items()}
