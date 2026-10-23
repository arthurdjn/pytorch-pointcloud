"""Frozen CLIP text encoder, the vocabulary side of open-vocabulary segmentation."""

import logging
from typing import TYPE_CHECKING, Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.utils.imports import _TRANSFORMERS_GITHUB_URL, optional_import

if TYPE_CHECKING:
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
else:
    CLIPTextModelWithProjection, _ = optional_import(
        "transformers", "CLIPTextModelWithProjection", url=_TRANSFORMERS_GITHUB_URL
    )
    CLIPTokenizer, _ = optional_import("transformers", "CLIPTokenizer", url=_TRANSFORMERS_GITHUB_URL)


class CLIPTextEncoder(nn.Module):
    r"""Frozen CLIP text tower embedding prompts into the space an open-vocabulary model is aligned with.

    The text model and its projection are read from the Hugging Face Hub through `transformers` (the vision tower is
    not instantiated) and frozen. Prompts are tokenized, encoded and projected, and the embeddings are normalized to
    unit length, so their dot product with normalized point embeddings is a cosine similarity.

    Args:
        name: Hub id of a CLIP checkpoint, e.g. `openai/clip-vit-large-patch14-336` (what OpenSeg and the OpenScene
            `openseg` weights use) or `openai/clip-vit-base-patch32` (LSeg).

    Example:
        ```{.python notest}
        from torch_pointcloud.layers import CLIPTextEncoder

        text_encoder = CLIPTextEncoder("openai/clip-vit-large-patch14-336")
        text = text_encoder(["a chair in a scene", "a table in a scene"])  # (2, 768), unit norm
        ```
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        # The checkpoint also holds the vision tower, which `transformers` reports as unexpected keys at warning level.
        logger = logging.getLogger("transformers.modeling_utils")
        level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            model = CLIPTextModelWithProjection.from_pretrained(name)
        finally:
            logger.setLevel(level)
        self.model: nn.Module = model
        self.model.train(False)
        self.model.requires_grad_(False)
        projection_dim = model.config.projection_dim
        assert projection_dim is not None
        self._embed_dim = int(projection_dim)

    @property
    def embed_dim(self) -> int:
        """Width $D$ of the projected text embeddings."""
        return self._embed_dim

    def forward(self, texts: Sequence[str]) -> Tensor:
        r"""Embed `texts` into unit vectors of shape $(K, D)$, truncating prompts longer than the context of CLIP."""
        tokens = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        embeddings = self.model(**tokens).text_embeds
        return F.normalize(embeddings, dim=-1)
