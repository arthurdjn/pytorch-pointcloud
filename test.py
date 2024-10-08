from typing import Any, Literal, Tuple, overload

import torch


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
    mode: str,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified scatter_sum function for both padded and packed modes.
    Depending on the mode, either 'lengths' (padded) or 'batch_idxs' (packed) must be passed as kwargs.
    """
    if mode == "auto":
        # Automatically detect the mode based on the shape of points
        if points.dim() == 3:  # (B, N, C) -> Padded mode
            mode = "padded"
            if "lengths" not in kwargs:
                raise ValueError("In 'auto' mode for padded format, 'lengths' must be provided.")

        elif points.dim() == 2:  # (N_total, C) -> Packed mode
            mode = "packed"
            if "batch_idxs" not in kwargs:
                raise ValueError("In 'auto' mode for packed format, 'batch_idxs' must be provided.")
        else:
            raise ValueError(f"Unsupported shape {points.shape} for auto mode. Expecting 2D or 3D tensor.")

    if mode == "padded":
        # Expecting 'lengths' in kwargs
        lengths = kwargs.get("lengths", None)
        if lengths is None:
            raise ValueError("In 'padded' mode, 'lengths' must be provided.")
        return points, cluster_ids

    elif mode == "packed":
        # Expecting 'batch_idxs' in kwargs
        batch_idxs = kwargs.get("batch_idxs")
        if batch_idxs is None:
            raise ValueError("In 'packed' mode, 'batch_idxs' must be provided.")
        return points, cluster_ids

    else:
        raise ValueError("Invalid mode. Expected 'padded' or 'packed'.")


points = torch.rand(10, 3)
cluster_ids = torch.randint(0, 5, (10,))
lengths = torch.tensor([2, 3, 5])
batch_idxs = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])


a = scatter_sum(points, cluster_ids, mode="padded", lengths=lengths)
a = scatter_sum(points, cluster_ids, mode="packed", batch_idxs=batch_idxs)
a = scatter_sum(points, cluster_ids, mode="auto", lengths=lengths)
a = scatter_sum(points, cluster_ids, mode="auto", batch_idxs=batch_idxs)
# a = scatter_sum(points, cluster_ids, mode="auto", lengths=lengths, batch_idxs=batch_idxs)  # Error
