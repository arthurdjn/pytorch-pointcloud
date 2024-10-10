from typing import Any, Literal, Tuple, overload

import torch

Mode = Literal["auto", "padded", "packed"]
StrictMode = Literal["padded", "packed"]


def resolve_strict_mode_params(dim: int, mode: str, **kwargs: Any) -> Tuple[StrictMode, Any, Any]:
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in ("lengths", "batch_idxs")}
    if extra_kwargs:
        extra_kwargs_str = ", ".join([f"{k}={v}" for k, v in extra_kwargs.items()])
        raise TypeError(f"Unexpected keyword arguments: {extra_kwargs_str}")

    if mode == "auto" and dim == 3:
        if "lengths" in kwargs:
            return "padded", kwargs["lengths"], None
        elif "batch_idxs" in kwargs:
            print("Warning: found 'batch_idxs' in kwargs for 'padded' mode. Converting it to 'lengths'.")
            lengths = torch.bincount(kwargs["batch_idxs"])
            return "padded", lengths, None
        return "padded", None, None

    elif mode == "auto" and dim == 2:
        if "batch_idxs" in kwargs:
            return "packed", None, kwargs["batch_idxs"]
        elif "lengths" in kwargs:
            print("Warning: found 'lengths' in kwargs for 'packed' mode. Converting it to 'batch_idxs'.")
            batch_idxs = torch.repeat_interleave(torch.arange(len(kwargs["lengths"])), kwargs["lengths"])
            return "packed", None, batch_idxs
        return "packed", None, None

    elif mode == "auto" and dim not in (2, 3):
        raise ValueError(f"Unsupported shape {dim}. Expecting 2D or 3D tensor.")

    elif mode == "padded":
        if "lengths" in kwargs:
            return "padded", kwargs["lengths"], None
        elif "batch_idxs" in kwargs:
            print("Warning: found 'batch_idxs' in kwargs for 'padded' mode. Converting it to 'lengths'.")
            lengths = torch.bincount(kwargs["batch_idxs"])
            return "padded", lengths, None
        return "padded", None, None

    elif mode == "packed":
        if "batch_idxs" in kwargs:
            return "packed", None, kwargs["batch_idxs"]
        elif "lengths" in kwargs:
            print("Warning: found 'lengths' in kwargs for 'packed' mode. Converting it to 'batch_idxs'.")
            batch_idxs = torch.repeat_interleave(torch.arange(len(kwargs["lengths"])), kwargs["lengths"])
            return "packed", None, batch_idxs
        return "packed", None, None

    raise ValueError("Invalid mode. Expected 'auto', 'padded', or 'packed'.")


@overload
def scatter_sum(
    points: torch.Tensor, cluster_ids: torch.Tensor, *, mode: Literal["auto"], lengths: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor]: ...


@overload
def scatter_sum(
    points: torch.Tensor, cluster_ids: torch.Tensor, *, mode: Literal["auto"], batch_idxs: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor]: ...


@overload
def scatter_sum(
    points: torch.Tensor, cluster_ids: torch.Tensor, *, mode: Literal["padded"], lengths: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor]: ...


@overload
def scatter_sum(
    points: torch.Tensor, cluster_ids: torch.Tensor, *, mode: Literal["packed"], batch_idxs: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def scatter_sum(
    points: torch.Tensor,
    cluster_ids: torch.Tensor,
    *,
    mode: Mode,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified scatter_sum function for both padded and packed modes.
    Depending on the mode, either 'lengths' (padded) or 'batch_idxs' (packed) can be passed as kwargs.
    """
    mode, lengths, batch_idxs = resolve_strict_mode_params(points.dim(), mode, **kwargs)
    if mode == "padded":
        return points, cluster_ids
    else:
        return points, cluster_ids


points = torch.rand(10, 3)
cluster_ids = torch.randint(0, 5, (10,))
lengths = torch.tensor([2, 3, 5])
batch_idxs = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])


a = scatter_sum(points, cluster_ids, mode="padded", lengths=lengths)
a = scatter_sum(points, cluster_ids, mode="packed", batch_idxs=batch_idxs)
a = scatter_sum(points, cluster_ids, mode="auto", lengths=lengths)
a = scatter_sum(points, cluster_ids, mode="auto", batch_idxs=batch_idxs)
# a = scatter_sum(points, cluster_ids, mode="auto", lengths=lengths, batch_idxs=batch_idxs)  # Error

scatter_sum(points, cluster_ids, mode="auto", batch_idxs=batch_idxs)  # Warning
