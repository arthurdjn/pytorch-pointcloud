r"""Nearest-neighbour refinement of part labels on top of another inferer's output.

Takes the per-point argmax of a base inferer and re-assigns the labels that are implausible for the shape
(rare parts, parts the shape's category does not own) by a majority vote of each point's nearest neighbours.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from torch_pointcloud.datasets.shapenetpart import ShapeNetPart
from torch_pointcloud.utils.data import DataKeys

from .inferer import Inferer
from .simple import SimpleInferer


def part_refinement_inference(
    data: Dict[str, Any],
    *,
    predictor: Callable[[Dict[str, Any]], Tensor],
    base: Optional[Inferer] = None,
    part_ids: Optional[Sequence[Sequence[int]]] = None,
    min_count: int = 10,
    num_neighbors: int = 11,
    category_key: str = DataKeys.CATEGORY,
    pos_key: str = DataKeys.POS,
    batch_key: str = DataKeys.BATCH,
) -> Tensor:
    r"""Nearest-neighbour refinement of part labels on top of another inferer.

    Runs `base`, takes the per-point argmax, and for every shape re-assigns the labels that are implausible:
    a predicted part with fewer than `min_count` points, or a part the shape's category does not own. Each
    such label is refined in turn (ascending label order): its points take the majority label of their
    `num_neighbors` nearest points of the same shape, the label under refinement excluded from the vote,
    with the already-refined labels feeding the next votes. This is the post-processing of the
    :arxiv: [PointNeXt](https://arxiv.org/abs/2206.04670) ShapeNetPart protocol.

    Args:
        data: Dict of per-point tensors. Must contain `pos` (shape $(N, D)$), `batch` (shape $(N,)$) and the
            per-shape category under `category_key`.
        predictor: Callable mapping a data dict to per-point part scores of shape $(N, C)$.
        base: Inferer running `predictor`; defaults to `SimpleInferer` (one forward on the whole batch).
        part_ids: Part labels owned by each category; defaults to the 16-category / 50-part ShapeNetPart table
            (`ShapeNetPart.seg_ids`).
        min_count: Predicted parts with fewer points than this are refined.
        num_neighbors: Number of nearest neighbours (the point itself included) voting on the new label.
        category_key: Dict key of the per-shape category, one-hot $(B, K)$ or index $(B,)$.
        pos_key: Dict key for the position tensor.
        batch_key: Dict key for the per-point batch index.

    Returns:
        One-hot refined labels of shape $(N, C)$, so a metric's argmax recovers the refined labels. An empty
        scene ($N = 0$) returns the base output unchanged.
    """
    for key in (pos_key, batch_key, category_key):
        if key not in data:
            raise KeyError(f"`data` is missing the required key {key!r}.")
    if min_count < 1:
        raise ValueError(f"`min_count` must be >= 1, got {min_count}.")
    if num_neighbors < 1:
        raise ValueError(f"`num_neighbors` must be >= 1, got {num_neighbors}.")

    base = base if base is not None else SimpleInferer()
    parts = [list(ids) for ids in (part_ids if part_ids is not None else ShapeNetPart.seg_ids.values())]

    scores = base(data, predictor)
    if scores.numel() == 0:
        return scores

    num_classes = int(scores.size(1))
    labels = scores.argmax(dim=1)
    pos = data[pos_key]
    batch = data[batch_key]
    category = data[category_key]
    if category.dim() == 2:
        category = category.argmax(dim=1)

    category = category.long()

    for b in torch.unique(batch).tolist():
        idx_b = torch.where(batch == b)[0]
        labels_b = labels[idx_b]
        counts = torch.bincount(labels_b, minlength=num_classes)
        present: List[int] = torch.where(counts > 0)[0].tolist()
        if len(present) <= 1:
            continue

        owned = set(parts[int(category[b])])
        pos_b = pos[idx_b]
        k = min(num_neighbors, int(idx_b.numel()))
        # Which points to refine is decided on the base predictions; the votes read the refined labels.
        initial = labels_b.clone()
        for label in present:
            if int(counts[label]) >= min_count and label in owned:
                continue

            rows = torch.where(initial == label)[0]
            neighbors = torch.cdist(pos_b[rows], pos_b).topk(k, dim=1, largest=False).indices  # (M, k)
            votes = torch.nn.functional.one_hot(labels_b[neighbors], num_classes=num_classes).sum(dim=1)
            votes[:, label] = 0
            # A row whose neighbours all carry `label` has no votes left; keep its label instead of
            # letting argmax fall through to class 0.
            has_votes = votes.sum(dim=1) > 0
            labels_b[rows[has_votes]] = votes.argmax(dim=1)[has_votes]

        labels[idx_b] = labels_b

    return torch.nn.functional.one_hot(labels, num_classes=num_classes).to(scores.dtype)


class PartRefinementInferer(Inferer):
    r"""Nearest-neighbour refinement of part labels on top of another inferer.

    Takes the base inferer's per-point argmax and re-assigns the implausible part labels (rare parts, parts
    the shape's category does not own) by a nearest-neighbour majority vote, returning one-hot scores.

    All parameters are forwarded verbatim to `part_refinement_inference`.

    Example:
        ```{.python notest}
        from torch_pointcloud.inferers import PartRefinementInferer, SimpleInferer

        inferer = PartRefinementInferer(SimpleInferer())
        scores = inferer(shapes, predictor=lambda d: model(d["x"], d["pos"], d["batch"], d["category"]))
        labels = scores.argmax(dim=1)
        ```
    """

    def __init__(
        self,
        base: Inferer,
        part_ids: Optional[Sequence[Sequence[int]]] = None,
        min_count: int = 10,
        num_neighbors: int = 11,
        category_key: str = DataKeys.CATEGORY,
        pos_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
    ) -> None:
        self.base = base
        self.part_ids = part_ids
        self.min_count = min_count
        self.num_neighbors = num_neighbors
        self.category_key = category_key
        self.pos_key = pos_key
        self.batch_key = batch_key

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        return part_refinement_inference(
            data,
            predictor=predictor,
            base=self.base,
            part_ids=self.part_ids,
            min_count=self.min_count,
            num_neighbors=self.num_neighbors,
            category_key=self.category_key,
            pos_key=self.pos_key,
            batch_key=self.batch_key,
        )
