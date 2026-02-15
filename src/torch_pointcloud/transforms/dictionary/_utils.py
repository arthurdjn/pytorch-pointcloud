from typing import Any, Dict, Generator, Iterable

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.types import KeyCollection


def assert_keys_in_data(
    data: Dict[str, Any],
    *keys: str,
    extra_msg: str = "",
) -> None:
    missing_keys = []
    for key in keys:
        if key not in data:
            missing_keys.append(key)

    if missing_keys:
        raise KeyError(
            f"Found {len(missing_keys)} missing keys in input data: {', '.join([f'{k!r}' for k in missing_keys])} "
            f"(available keys are: {', '.join(data.keys())}) {extra_msg}"
        )


def key_iterator(
    data: Dict[str, Any],
    keys: KeyCollection,
    *extra_iterables: Iterable[Any],
    allow_missing_keys: bool = False,
    extra_msg: str = "",
) -> Generator[Any, None, None]:
    # inspired by: https://github.com/Project-MONAI/MONAI/blob/main/monai/transforms/transform.py#L456
    # if no extra iterables given, create a dummy list of Nones
    ex_iters: Iterable[Any] = extra_iterables or [[None] * len(keys)]
    ex_iters = [ensure_tuple(ex_iter) for ex_iter in ex_iters]

    for key, *_ex_iters in zip(keys, *ex_iters):
        if key in data:
            # all normal, yield (what we yield depends on whether extra iterables were given)
            yield (key,) + tuple(_ex_iters) if extra_iterables else key
        elif not allow_missing_keys:
            raise KeyError(f"Key {key!r} was missing in the data and `allow_missing_keys==False`. {extra_msg}")
