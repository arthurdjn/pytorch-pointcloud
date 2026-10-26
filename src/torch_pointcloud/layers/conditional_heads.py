"""Classification heads holding one class table per condition, selected at forward time."""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _condition_index(conditions: Tuple[str, ...], condition: Optional[str]) -> int:
    if condition is None:
        options = ", ".join(repr(c) for c in conditions)
        raise ValueError(f"The head requires a condition. Valid conditions are: {options}.")

    if condition not in conditions:
        options = ", ".join(repr(c) for c in conditions)
        raise ValueError(f"Unknown condition {condition!r}. Valid conditions are: {options}.")

    return conditions.index(condition)


# inspired by: https://github.com/Pointcept/Pointcept/blob/v1.5.1/pointcept/models/point_prompt_training/point_prompt_training_v1m1_language_guided.py#L19
class CosineSimilarityHead(nn.Module):
    r"""Classifier scoring features by cosine similarity against fixed class embeddings, with one class table per
    condition.

    The features are projected into the embedding space, $\ell_2$-normalized and scored against the class table $E_c$
    of the active condition, times a fixed scale $\alpha$:
    $\text{logits} = \alpha \, \hat{z} E_c^\top$ with $\hat{z} = \text{proj}(x) / \lVert \text{proj}(x) \rVert$.
    Only the projection is learned. The class embeddings, typically text embeddings of the class names computed once
    by a text encoder, are the `class_embedding` buffer stored with the weights; the rows of condition $i$ follow
    those of conditions $0, \ldots, i - 1$. `from_embeddings` builds the head from given embeddings, while the
    constructor fills the buffer with random unit vectors for a state dict to replace.

    Introduced in :arxiv: [Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt
    Training](https://arxiv.org/abs/2308.09718)

    Args:
        in_channels: Width $C$ of the input features.
        embed_dim: Width $D$ of the embedding space.
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
        from torch_pointcloud.layers import CosineSimilarityHead

        class_embedding = torch.randn(13, 512)  # one embedding per class name
        head = CosineSimilarityHead.from_embeddings(64, class_embedding, scale=100.0)
        logits = head(torch.randn(32, 64))
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

    @classmethod
    def from_embeddings(
        cls,
        in_channels: int,
        class_embedding: Union[Tensor, Sequence[Tensor]],
        scale: float,
        conditions: Optional[Sequence[str]] = None,
    ) -> "CosineSimilarityHead":
        r"""Build a head whose class tables are the given embeddings, $\ell_2$-normalized.

        Args:
            in_channels: Width $C$ of the input features.
            class_embedding: One $(K, D)$ table, or one $(K_i, D)$ table per condition when `conditions` is given.
            scale: Scale $\alpha$ of the cosine similarities.
            conditions: Ordered condition names, one per table.

        Returns:
            The head, with `num_classes` and `embed_dim` read from the tables.

        Example:
            ```python
            import torch
            from torch_pointcloud.layers import CosineSimilarityHead

            tables = [torch.randn(20, 512), torch.randn(13, 512)]
            head = CosineSimilarityHead.from_embeddings(64, tables, scale=100.0, conditions=["indoor", "office"])
            logits = head(torch.randn(32, 64), condition="office")
            print(logits.shape)  # torch.Size([32, 13])
            ```
        """
        if isinstance(class_embedding, Tensor):
            tables = [class_embedding]
            num_classes: Union[int, Tuple[int, ...]] = class_embedding.shape[0]
        else:
            tables = list(class_embedding)
            num_classes = tuple(table.shape[0] for table in tables)

        head = cls(in_channels, tables[0].shape[1], num_classes, scale, conditions=conditions)
        head.class_embedding.copy_(F.normalize(torch.cat(tables).detach(), dim=1))
        return head

    def forward(self, x: Tensor, condition: Optional[str] = None) -> Tensor:
        start, num_classes = 0, self.num_classes[0]
        if self.conditions is not None:
            index = _condition_index(self.conditions, condition)
            start, num_classes = sum(self.num_classes[:index]), self.num_classes[index]

        z = F.normalize(self.proj(x), dim=1)
        logits = self.scale * F.linear(z, self.class_embedding[start : start + num_classes])
        return logits


# inspired by: https://github.com/Pointcept/Pointcept/blob/v1.5.1/pointcept/models/point_prompt_training/point_prompt_training_v1m2_decoupled.py#L44
class ConditionalLinear(nn.ModuleList):
    r"""Linear classifier with one independent `nn.Linear` per condition, selected at forward time.

    Each condition keeps its own label set: the layer of condition $i$ maps the $C$ input channels to its $K_i$
    classes. The layers are the children `0, ..., n - 1` of this `nn.ModuleList`, in the order of `conditions`.

    Introduced in :arxiv: [Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt
    Training](https://arxiv.org/abs/2308.09718)

    Args:
        in_channels: Width $C$ of the input features.
        num_classes: One class count $K_i$ per condition.
        conditions: Ordered condition names, one per class count.

    Shape:
        - Input: $(N, C)$ packed features.
        - Output: $(N, K_c)$ logits, with $K_c$ the class count of the selected condition.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import ConditionalLinear

        head = ConditionalLinear(64, num_classes=(20, 13), conditions=["indoor", "office"])
        logits = head(torch.randn(32, 64), condition="office")
        print(logits.shape)  # torch.Size([32, 13])
        ```
    """

    def __init__(self, in_channels: int, num_classes: Sequence[int], conditions: Sequence[str]) -> None:
        if len(num_classes) != len(conditions):
            raise ValueError(
                f"`num_classes` has {len(num_classes)} entries; `conditions` must name as many (got {list(conditions)!r})."
            )

        super().__init__([nn.Linear(in_channels, k) for k in num_classes])
        self.in_channels = in_channels
        self.num_classes = tuple(num_classes)
        self.conditions = tuple(conditions)

    def forward(self, x: Tensor, condition: Optional[str] = None) -> Tensor:
        linear = self[_condition_index(self.conditions, condition)]
        logits: Tensor = linear(x)
        return logits
