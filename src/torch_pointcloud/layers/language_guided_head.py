"""Language-guided classification head scoring features against fixed class-text embeddings."""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LanguageGuidedHead(nn.Module):
    r"""Classifier whose class weights are fixed text embeddings of the class names, with one class table per
    condition.

    The features are projected into the text-embedding space, $\ell_2$-normalized and scored by cosine similarity
    against the class table $E_c$ of the active condition, times a fixed scale $\alpha$:
    $\text{logits} = \alpha \, \hat{z} E_c^\top$ with $\hat{z} = \text{proj}(x) / \lVert \text{proj}(x) \rVert$.
    Only the projection is learned. The tables are computed once by a text encoder and stored with the weights in
    the `class_embedding` buffer, so no text encoder runs at inference: the rows of condition $i$ follow those of
    conditions $0, \ldots, i - 1$, in class order. A fresh head holds random unit vectors; copy the normalized text
    embeddings into `class_embedding` before training.

    Introduced in :arxiv: [Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt
    Training](https://arxiv.org/abs/2308.09718)

    Args:
        in_channels: Width $C$ of the input features.
        embed_dim: Width $D$ of the text-embedding space.
        num_classes: Number of classes, or one count per condition when `conditions` is given.
        scale: Scale $\alpha$ of the cosine similarities.
        conditions: Ordered condition names selecting the class table at forward time. `None` builds a single
            table and ignores the `condition` argument.

    Shape:
        - Input: $(N, C)$ packed features.
        - Output: $(N, K_c)$ logits, with $K_c$ the class count of the selected condition.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import LanguageGuidedHead

        head = LanguageGuidedHead(64, 512, num_classes=(20, 13), scale=100.0, conditions=["indoor", "office"])
        x = torch.randn(32, 64)
        logits = head(x, condition="office")
        print(logits.shape)  # torch.Size([32, 13])
        ```
    """

    class_embedding: Tensor

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_classes: Union[int, Sequence[int]],
        scale: float,
        conditions: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, int) and conditions is not None:
            raise ValueError(
                f"Got conditions={list(conditions)!r} but a single `num_classes`; pass one count per condition."
            )
        if not isinstance(num_classes, int) and (conditions is None or len(conditions) != len(num_classes)):
            raise ValueError(
                f"`num_classes` has {len(num_classes)} entries; `conditions` must name as many "
                f"(got {None if conditions is None else list(conditions)!r})."
            )

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes: Tuple[int, ...] = (num_classes,) if isinstance(num_classes, int) else tuple(num_classes)
        self.scale = scale
        self.conditions = None if conditions is None else tuple(conditions)
        self.proj = nn.Linear(in_channels, embed_dim)
        self.register_buffer("class_embedding", F.normalize(torch.randn(sum(self.num_classes), embed_dim), dim=1))

    def forward(self, x: Tensor, condition: Optional[str] = None) -> Tensor:
        start, num_classes = 0, self.num_classes[0]
        if self.conditions is not None:
            if condition is None:
                options = ", ".join(repr(c) for c in self.conditions)
                raise ValueError(f"LanguageGuidedHead requires a condition. Valid conditions are: {options}.")

            if condition not in self.conditions:
                options = ", ".join(repr(c) for c in self.conditions)
                raise ValueError(f"Unknown condition {condition!r}. Valid conditions are: {options}.")

            index = self.conditions.index(condition)
            start, num_classes = sum(self.num_classes[:index]), self.num_classes[index]

        z = F.normalize(self.proj(x), dim=1)
        logits = self.scale * F.linear(z, self.class_embedding[start : start + num_classes])
        return logits
