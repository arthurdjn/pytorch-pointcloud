"""Transforms that rename, copy, convert or combine the entries of a sample dict, and element-wise arithmetic."""

from typing import Any, Dict, Literal, Optional, Sequence, Union, get_args

import numpy as np
import torch
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_tuple_size
from torch_pointcloud.utils.types import KeyCollection, ValueCollection

from .base import DictTransform

ReduceOp = Literal["min", "max", "mean", "sum"]
"""Allowed values for `Reduce.op` (per-key reduction operator)."""

__all__ = [
    "Abs",
    "Cat",
    "Clamp",
    "CopyItems",
    "Divide",
    "DivideItems",
    "KeepItems",
    "OneHot",
    "OnesLike",
    "Reduce",
    "Relabel",
    "RenameItems",
    "Scale",
    "SetValue",
    "SubtractItems",
    "ToDevice",
    "ToFloat",
    "ToTensor",
]


class SetValue(DictTransform):
    """Set values for keys in the dictionary, creating or overwriting them.

    Unlike most `DictTransform` subclasses, `SetValue` does not read existing
    values, so `allow_missing_keys` has no meaning and is not accepted.

    ![SetValue diagram](../../assets/transforms/set_value.png)

    Args:
        keys: The keys to set.
        values: The values to set. Either a single value broadcast to every key,
            or a sequence of values the same length as `keys`.
    """

    def __init__(self, keys: KeyCollection, values: Any) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.values = ensure_tuple_size(values, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, value in zip(self.keys, self.values):
            data[key] = value
        return data


class ToFloat(DictTransform):
    """Cast dictionary tensor entries to float32.

    Useful when tensors are stored in integer formats (e.g. `uint8` for
    colors) and need to be promoted to floating point before arithmetic
    transforms like `Divide` or `Normalize`.

    ![ToFloat diagram](../../assets/transforms/to_float.png)

    Args:
        keys: The keys to cast.
        allow_missing_keys: If `True`, missing keys are silently ignored.
    """

    def __init__(
        self,
        keys: KeyCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key in self.iter_keys(data):
            data[key] = data[key].float()
        return data


class ToDevice(DictTransform):
    """Convert dictionary tensor entries to the given device.

    ![ToDevice diagram](../../assets/transforms/to_device.png)

    Args:
        keys: The keys to convert the tensors to the given device.
        device: The device to convert the tensors to.
        non_blocking: If `True`, the transfer will be done asynchronously.
        copy: If `True`, the tensor will be copied to the new device.
        memory_format: The memory format to use for the tensor.
        dst_keys: The keys to store the converted tensors in.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        device: ValueCollection[str | torch.device],
        non_blocking: ValueCollection[bool] = False,
        copy: ValueCollection[bool] = True,
        memory_format: ValueCollection[torch.memory_format | None] = None,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.device = ensure_tuple_size(device, len(self.keys))
        self.non_blocking = ensure_tuple_size(non_blocking, len(self.keys))
        self.copy = ensure_tuple_size(copy, len(self.keys))
        self.memory_format = ensure_tuple_size(memory_format, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, device, non_blocking, copy, memory_format in self.iter_keys(
            data,
            self.dst_keys,
            self.device,
            self.non_blocking,
            self.copy,
            self.memory_format,
        ):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")

            data[dst_key] = x.to(
                device,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            )

        return data


def relabel(
    labels: Tensor,
    mapping: Union[Sequence[int], Dict[int, int]],
    default: int = 0,
) -> Tensor:
    """Remap integer labels via a lookup table.

    `mapping` can be either:

    - a sequence of source values (1:1): each value at index $i$ is mapped to $i$;
    - a `dict[int, int]` (general source → target): supports N-to-1 merges
      (e.g. SemanticKITTI's `moving-car` and `car` both → 0).

    Source values not listed in `mapping` are set to `default`.

    Args:
        labels: Integer label tensor (any integer dtype). Output preserves dtype.
        mapping: Source-value listing (1:1) or explicit `{source: target}` dict (N:1).
        default: Value assigned to source values not listed in `mapping`.

    Returns:
        Remapped tensor with the same shape and dtype as `labels`.

    Raises:
        ValueError: If `mapping` is empty.
    """
    if isinstance(mapping, dict):
        table: Dict[int, int] = {int(k): int(v) for k, v in mapping.items()}
    else:
        table = {int(v): i for i, v in enumerate(mapping)}
    if not table:
        raise ValueError("relabel requires at least one source value in `mapping`.")
    sorted_sources = sorted(table.keys())
    src = torch.tensor(sorted_sources, dtype=torch.long, device=labels.device)
    tgt = torch.tensor([table[s] for s in sorted_sources], dtype=torch.long, device=labels.device)
    labels_long = labels.long()
    idx = torch.searchsorted(src, labels_long)
    idx_clamped = idx.clamp(max=src.numel() - 1)
    hit = src[idx_clamped] == labels_long
    dst = torch.full_like(labels_long, default)
    dst[hit] = tgt[idx_clamped[hit]]
    return dst.to(labels.dtype)


class Relabel(DictTransform):
    """Remap integer labels in dictionary entries via a lookup table.

    `labels` can be either:

    - a sequence of source values (1:1) - each value at index `i` is mapped to `i`;
    - a `dict[int, int]` (general source → target) - supports N-to-1 merges
      (e.g. SemanticKITTI's `moving-car` and `car` both → 0).

    Source values not listed in `labels` are set to `default`.

    ![Relabel before / after](../../assets/transforms/relabel.png)

    Args:
        keys: Keys holding label tensors to remap.
        labels: Source-value listing (1:1) or explicit `{source: target}` dict (N:1).
        default: Value assigned to source values not listed in `labels`.
        allow_missing_keys: If `True`, skip missing keys instead of raising.

    Example:
        ```python
        # 1:1 - keep raw NYU40 ids 1, 2, 3, 4, 5 and remap them to 0..4
        T.Relabel(keys="segment", labels=[1, 2, 3, 4, 5])

        # N:1 - SemanticKITTI 19-class benchmark (merges moving-* into static)
        T.Relabel(
            keys="segment",
            labels={
                10: 0, 252: 0,    # car        (+ moving-car)
                11: 1,             # bicycle
                15: 2,             # motorcycle
                18: 3, 258: 3,    # truck      (+ moving-truck)
                # ...
            },
            default=255,
        )
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        labels: Sequence[int] | Dict[int, int],
        default: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.default = default

        if isinstance(labels, dict):
            self.labels: Dict[int, int] = {int(k): int(v) for k, v in labels.items()}
        else:
            self.labels = {int(value): idx for idx, value in enumerate(labels)}

        if not self.labels:
            raise ValueError("Relabel requires at least one source value in `labels`.")

    def transform(self, data: dict) -> dict:
        data = dict(data)

        for key in self.iter_keys(data):
            tensor = data[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for key {key!r}, got {type(tensor).__name__}")
            data[key] = relabel(tensor, self.labels, default=self.default)

        return data


class RenameItems(DictTransform):
    """Rename keys in the dictionary.

    ![RenameItems diagram](../../assets/transforms/rename_items.png)

    Args:
        keys: Source keys to rename.
        dst_keys: New key names (same length as `keys`).
        allow_missing_keys: If `True`, silently skip absent source keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dst_keys: KeyCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = data.pop(key)
        return data


class CopyItems(DictTransform):
    """Copy values from source keys to new destination keys.

    ![CopyItems diagram](../../assets/transforms/copy_items.png)

    Args:
        keys: Source keys to copy from.
        dst_keys: Destination keys to copy to (same length as `keys`).
        allow_missing_keys: If `True`, silently skip absent source keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dst_keys: KeyCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            val = data[key]
            if torch.is_tensor(val):
                val = val.clone()
            elif isinstance(val, np.ndarray):
                val = val.copy()
            data[dst_key] = val
        return data


class ToTensor(DictTransform):
    """Convert dictionary entries to tensors.

    ![ToTensor diagram](../../assets/transforms/to_tensor.png)

    Args:
        keys: The keys to convert.
        dtype: Target dtype(s).
        device: Target device(s).
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dtype: ValueCollection[str | torch.dtype] | None = None,
        device: ValueCollection[str | torch.device] | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dtype = ensure_tuple_size(dtype, len(self.keys))
        self.device = ensure_tuple_size(device, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dtype, device in self.iter_keys(data, self.dtype, self.device):
            data[key] = torch.as_tensor(data[key], dtype=dtype, device=device)
        return data


class OnesLike(DictTransform):
    """Adds a tensor of ones shaped like existing dictionary entries.

    ![OnesLike diagram](../../assets/transforms/ones_like.png)

    Args:
        keys: Reference keys used to determine tensor shape.
        dst_keys: Keys under which the ones tensors are stored.
    """

    def __init__(
        self,
        keys: KeyCollection,
        memory_format: ValueCollection[torch.memory_format] | None = None,
        dtype: ValueCollection[torch.dtype] | None = None,
        layout: ValueCollection[torch.layout] | None = None,
        device: ValueCollection[torch.device] | None = None,
        pin_memory: ValueCollection[bool] | None = False,
        requires_grad: ValueCollection[bool] | None = False,
        dst_keys: KeyCollection | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.memory_format = ensure_tuple_size(memory_format, len(self.keys))
        self.dtype = ensure_tuple_size(dtype, len(self.keys))
        self.layout = ensure_tuple_size(layout, len(self.keys))
        self.device = ensure_tuple_size(device, len(self.keys))
        self.pin_memory = ensure_tuple_size(pin_memory, len(self.keys))
        self.requires_grad = ensure_tuple_size(requires_grad, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, memory_format, dtype, layout, device, pin_memory, requires_grad in self.iter_keys(
            data,
            self.dst_keys,
            self.memory_format,
            self.dtype,
            self.layout,
            self.device,
            self.pin_memory,
            self.requires_grad,
        ):
            data[dst_key] = torch.ones_like(
                data[key],
                memory_format=memory_format,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory,
                requires_grad=requires_grad,
            )
        return data


class Cat(DictTransform):
    """Concatenates tensors from multiple keys into a single feature tensor.

    Note:
        This transform is mostly used to concatenate multiple features into a single tensor to feed into your model.

    Integer inputs are cast to `float32`; floating inputs keep their dtype. When the inputs mix
    floating dtypes, the result uses the widest one (so `float64` is preserved, never downcast).

    ![Cat diagram](../../assets/transforms/cat.png)

    Args:
        keys: Keys whose tensors are concatenated (in order).
        dst_key: Key under which the result is stored.
        dim: Dimension along which to concatenate.
        allow_missing_keys: If `True`, silently skip absent keys.

    Example:
        If you have a point cloud data containing position, color and normal and want to concatenate them
        into a single feature tensor (to feed into your model), you can do the following:

        ```python
        from torch_pointcloud.transforms import Cat

        data = {
            "pos": torch.randn(10, 3),
            "color": torch.randn(10, 3),
            "normal": torch.randn(10, 3),
        }
        transform = Cat(keys=["pos", "color", "normal"], dst_key="x", dim=1)
        data = transform(data)
        ```

        Now, the data dictionary will contain the key `x` with the shape $(10, 9)$.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dst_key: str,
        dim: int = -1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_key = dst_key
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        tensors = [data[key] if data[key].is_floating_point() else data[key].float() for key in self.iter_keys(data)]
        if not tensors:
            return data
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        data[self.dst_key] = torch.cat([tensor.to(dtype) for tensor in tensors], dim=self.dim)
        return data


class OneHot(DictTransform):
    r"""One-hot encode integer-class tensors.

    Wraps `torch.nn.functional.one_hot` and casts the result to float so the
    output is ready to feed into a model.

    ![OneHot diagram](../../assets/transforms/one_hot.png)

    Args:
        keys: Keys holding integer (long) class indices.
        num_classes: Number of classes $C$ in the one-hot encoding.
        dst_keys: Where to store the one-hot tensors. Defaults to `keys`.
        allow_missing_keys: If `True`, silently skip absent keys.

    Shape:
        Input class tensor of shape $(N,)$ becomes $(N, C)$. A scalar input
        becomes shape $(C,)$, which after batched collate stacks to $(B, C)$.
    """

    def __init__(
        self,
        keys: KeyCollection,
        num_classes: ValueCollection[int],
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.num_classes = ensure_tuple_size(num_classes, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, num_classes in self.iter_keys(data, self.dst_keys, self.num_classes):
            x = data[key].long()
            out = torch.nn.functional.one_hot(x, num_classes=num_classes).float()
            # A 0-d input (per-sample scalar label) one-hots to `(num_classes,)`. Unsqueeze
            # so packed-batch collate yields `(B, num_classes)` after `torch.cat(dim=0)`.
            if x.ndim == 0:
                out = out.unsqueeze(0)
            data[dst_key] = out
        return data


class Reduce(DictTransform):
    r"""Reduce a tensor along a dimension and store the scalar/vector result.

    Useful for capturing per-sample statistics (e.g. axis-wise scene maxima or
    centroids) as standalone keys that downstream transforms can reference.

    ![Reduce diagram](../../assets/transforms/reduce.png)

    Args:
        keys: Keys to reduce.
        op: Reduction operator: `"min"`, `"max"`, `"mean"`, or `"sum"` (matches the
            vocabulary used by `Voxelize`). `"mean"` keeps the input's floating dtype
            (`float64` included); integer inputs are cast to `float32`.
        dim: Dimension to reduce. Defaults to `0`.
        keepdim: Pass `keepdim=True` to keep the reduced axis as size $1$. This
            is helpful when the result is meant to broadcast against a $(N, D)$
            tensor (e.g. per-sample bbox stats) and to survive the packed-batch
            collate - a $(1, D)$ tensor collates to $(B, D)$ via `torch.cat`,
            whereas a $(D,)$ tensor would concatenate to $(B \cdot D,)$.
        dst_keys: Output keys. Defaults to `keys` (in-place overwrite).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    _OP_FUNCS: Dict[str, Any] = {
        "min": torch.amin,
        "max": torch.amax,
        "sum": torch.sum,
    }

    def __init__(
        self,
        keys: KeyCollection,
        op: ValueCollection[ReduceOp],
        dim: ValueCollection[int] = 0,
        keepdim: bool = False,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.op = ensure_tuple_size(op, len(self.keys))
        self.dim = ensure_tuple_size(dim, len(self.keys))
        self.keepdim = keepdim

        valid = get_args(ReduceOp)
        invalid = set(self.op) - set(valid)
        if invalid:
            raise ValueError(f"Invalid op(s): {invalid}. Expected one of {valid}.")

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, op, dim in self.iter_keys(data, self.dst_keys, self.op, self.dim):
            x = data[key]
            if op == "mean":
                x_float = x if x.is_floating_point() else x.float()
                data[dst_key] = x_float.mean(dim=dim, keepdim=self.keepdim)
            else:
                data[dst_key] = self._OP_FUNCS[op](x, dim=dim, keepdim=self.keepdim)
        return data


class KeepItems(DictTransform):
    r"""Keep only items in the data dictionary that are in the keys list.

    Note:
        This transform is useful if during augmentation process you constructed multiple tensors and want
        to drop intermediate tensors for memory efficiency.

    ![KeepItems diagram](../../assets/transforms/keep_items.png)

    Args:
        keys: The keys to keep in the data dictionary.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Example:
        If you have a data dictionary containing position, color and normal and want to keep only the position and color,
        you can do the following:

        ```python
        from torch_pointcloud.transforms import KeepItems

        data = {
            "pos": torch.randn(10, 3),
            "color": torch.randn(10, 3),
            "normal": torch.randn(10, 3),
        }
        transform = KeepItems(keys=["pos", "color"])
        data = transform(data)
        ```

        Now, the data dictionary will contain only the keys `pos` and `color`.
        The key `normal` will be removed.
    """

    def __init__(
        self,
        keys: KeyCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        return {key: data[key] for key in self.iter_keys(data)}


class Scale(DictTransform):
    """Multiply dictionary tensor entries by a scale factor.

    === "Object"

        ![Scale on an object](../../assets/transforms/scale.png)

    === "Scene"

        ![Scale on a room](../../assets/transforms/scale_scene.png)

    Args:
        keys: The keys to scale.
        scale: The scale factor(s).
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        scale: float | Sequence[float],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scale = ensure_tuple_size(scale, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, scale in self.iter_keys(data, self.scale):
            data[key] = data[key] * scale
        return data


class Divide(DictTransform):
    """Divide dictionary tensor entries by a divisor.

    === "Object"

        ![Divide on an object](../../assets/transforms/divide.png)

    === "Scene"

        ![Divide on a room](../../assets/transforms/divide_scene.png)

    Args:
        keys: The keys to divide.
        divisor: The divisor(s).
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        divisor: float | Sequence[float],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.divisor = ensure_tuple_size(divisor, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, divisor in self.iter_keys(data, self.divisor):
            data[key] = data[key] / divisor
        return data


class DivideItems(DictTransform):
    """Divide target keys by the value of a reference key element-wise.

    Computes `data[key] = data[key] / data[div_key]` for each key.

    ![DivideItems diagram](../../assets/transforms/divide_items.png)

    Args:
        keys: Keys whose tensors are divided.
        div_keys: Keys whose values are used as the divisors.
        dst_keys: Where to store results. Defaults to `keys`.
        allow_missing_keys: If `True`, silently skip absent target keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        div_keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.div_keys = ensure_tuple_size(div_keys, len(self.keys))
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, div_key, dst_key in self.iter_keys(data, self.div_keys, self.dst_keys):
            data[dst_key] = data[key] / data[div_key]
        return data


class SubtractItems(DictTransform):
    """Subtract the value of a reference key from target keys element-wise.

    Computes `data[key] = data[key] - data[sub_key]` for each key. With `axes`
    set, only the listed last-dim indices are subtracted; the other components
    pass through unchanged (useful to shift only XY while keeping Z absolute).

    === "Object"

        ![SubtractItems on an object](../../assets/transforms/subtract_items.png)

    === "Scene"

        ![SubtractItems on a room](../../assets/transforms/subtract_items_scene.png)

    Args:
        keys: Keys whose tensors are modified (subtracted from).
        sub_keys: Keys whose values are subtracted from each target key.
        dst_keys: Where to store results. Defaults to `keys`.
        axes: Optional indices into the last dim restricting which components are
            subtracted. `None` (default) subtracts every component.
        allow_missing_keys: If `True`, silently skip absent target keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        sub_keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        axes: Optional[Sequence[int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.sub_keys = ensure_tuple_size(sub_keys, len(self.keys))
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.axes = tuple(axes) if axes is not None else None

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, sub_key, dst_key in self.iter_keys(data, self.sub_keys, self.dst_keys):
            if self.axes is None:
                data[dst_key] = data[key] - data[sub_key]
            else:
                out = data[key].clone()
                idx = list(self.axes)
                out[..., idx] = out[..., idx] - data[sub_key][..., idx]
                data[dst_key] = out
        return data


def absolute(x: Tensor, inplace: bool = False) -> Tensor:
    """Make the input tensor absolute.

    Args:
        x: The input tensor.

    Returns:
        The absolute tensor.

    Examples:
        ```pycon
        >>> import torch
        >>> import torch_pointcloud.transforms.functional as F
        >>> x = torch.tensor([-1.0, 2.0, -3.0])
        >>> F.absolute(x)
        tensor([1., 2., 3.])

        ```
    """
    if inplace:
        x.abs_()
        return x

    return x.abs()


class Abs(DictTransform):
    """Make dictionary tensor entries absolute.

    See Also:
        `torch_pointcloud.transforms.functional.absolute`

    === "Object"

        ![Abs on an object](../../assets/transforms/abs.png)

    === "Scene"

        ![Abs on a room](../../assets/transforms/abs_scene.png)

    Args:
        keys: The keys to make absolute.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(self, keys: KeyCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.iter_keys(d):
            d[key] = absolute(d[key])
        return d


class Clamp(DictTransform):
    """Clamp tensor entries to a range (a thin wrapper over `torch.clamp`).

    === "Object"

        ![Clamp on an object](../../assets/transforms/clamp.png)

    === "Scene"

        ![Clamp on a room](../../assets/transforms/clamp_scene.png)

    Args:
        keys: Keys to clamp.
        min: Lower bound. `None` disables the lower clamp.
        max: Upper bound. `None` disables the upper clamp.
        dst_keys: Where to store the result. Defaults to `keys` (in-place overwrite).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        min: Optional[float] = None,
        max: Optional[float] = None,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if min is None and max is None:
            raise ValueError("Clamp requires at least one of `min` or `max`.")
        self.min = min
        self.max = max
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = data[key].clamp(min=self.min, max=self.max)
        return data
