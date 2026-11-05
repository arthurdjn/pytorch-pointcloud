"""Classification heads holding one class table per condition, selected at forward time."""

from typing import List, Optional, Sequence, Tuple, Union

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


def _class_counts(num_classes: Union[int, Sequence[int]], conditions: Optional[Sequence[str]]) -> Tuple[int, ...]:
    if not isinstance(num_classes, Sequence):
        if conditions is not None:
            raise ValueError(
                f"Got conditions={list(conditions)!r} but a single class count ({num_classes}); pass one per condition."
            )
        return (int(num_classes),)

    if conditions is None or len(conditions) != len(num_classes):
        raise ValueError(
            f"Got {len(num_classes)} class counts; `conditions` must name as many "
            f"(got {None if conditions is None else list(conditions)!r})."
        )

    return tuple(int(k) for k in num_classes)


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
    constructor fills the buffer with random unit vectors for a state dict to replace. `reset_class_embedding` swaps
    the tables of a trained head and keeps its projection: embeddings of new class names from the same text encoder
    give a classifier over those names without training.

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
        counts = _class_counts(num_classes, conditions)

        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = counts
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
        num_classes: Union[int, List[int]]
        if isinstance(class_embedding, Tensor):
            embed_dim, num_classes = class_embedding.shape[1], class_embedding.shape[0]
        else:
            embed_dim, num_classes = class_embedding[0].shape[1], [table.shape[0] for table in class_embedding]

        head = cls(in_channels, embed_dim, num_classes, scale, conditions=conditions)
        head.reset_class_embedding(class_embedding, conditions=conditions)
        return head

    def reset_class_embedding(
        self,
        class_embedding: Union[Tensor, Sequence[Tensor]],
        conditions: Optional[Sequence[str]] = None,
    ) -> None:
        r"""Replace the class tables by the given embeddings, $\ell_2$-normalized, keeping the projection.

        The head then scores the classes of the new tables. With embeddings from the text encoder the projection was
        trained against, this turns a trained head into a classifier over a new vocabulary.

        Args:
            class_embedding: One $(K, D)$ table, or one $(K_i, D)$ table per condition when `conditions` is given,
                with $D$ the `embed_dim` of the head.
            conditions: Ordered condition names, one per table.

        Example:
            ```python
            import torch
            from torch_pointcloud.layers import CosineSimilarityHead

            head = CosineSimilarityHead.from_embeddings(64, torch.randn(13, 512), scale=100.0)
            head.reset_class_embedding(torch.randn(5, 512))  # five new class names, same projection
            logits = head(torch.randn(32, 64))
            print(logits.shape)  # torch.Size([32, 5])
            ```
        """
        tables = [class_embedding] if isinstance(class_embedding, Tensor) else list(class_embedding)
        widths = sorted({table.shape[1] for table in tables})
        if widths != [self.embed_dim]:
            raise ValueError(f"The class tables must have the embedding width {self.embed_dim}, got widths {widths}.")
        num_classes: Union[int, List[int]]
        num_classes = tables[0].shape[0] if isinstance(class_embedding, Tensor) else [t.shape[0] for t in tables]
        counts = _class_counts(num_classes, conditions)

        self.num_classes = counts
        self.conditions = None if conditions is None else tuple(conditions)
        self.class_embedding = F.normalize(torch.cat(tables).detach().to(self.class_embedding), dim=1)

    def forward(self, x: Tensor, condition: Optional[str] = None) -> Tensor:
        start, num_classes = 0, self.num_classes[0]
        if self.conditions is not None:
            index = _condition_index(self.conditions, condition)
            start, num_classes = sum(self.num_classes[:index]), self.num_classes[index]

        z = F.normalize(self.proj(x), dim=1)
        logits = self.scale * F.linear(z, self.class_embedding[start : start + num_classes])
        return logits


# inspired by: https://github.com/Pointcept/Pointcept/blob/v1.5.1/pointcept/models/point_prompt_training/point_prompt_training_v1m2_decoupled.py#L44
class ConditionalLinear(nn.Module):
    r"""Linear classifier with one independent `nn.Linear` per condition, selected at forward time.

    Each condition keeps its own label set: the layer of condition $i$ maps the $C$ input channels to its $K_i$
    classes. The layers are `layers[0], ..., layers[n - 1]`, in the order of `conditions`.

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
        counts = _class_counts(num_classes, conditions)

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = counts
        self.conditions = tuple(conditions)
        self.layers = nn.ModuleList([nn.Linear(in_channels, k) for k in counts])

    def forward(self, x: Tensor, condition: Optional[str] = None) -> Tensor:
        linear = self.layers[_condition_index(self.conditions, condition)]
        logits: Tensor = linear(x)
        return logits
