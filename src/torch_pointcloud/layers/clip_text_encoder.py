"""Frozen CLIP text encoder, the vocabulary side of open-vocabulary segmentation."""

import logging
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

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
    not instantiated) and frozen. Prompts are tokenized, truncated to the context length of the model, encoded and
    projected, and the embeddings are normalized to unit length, so their product with unit point embeddings is a
    cosine similarity.

    Args:
        name: Hub id of a CLIP checkpoint (e.g. `openai/clip-vit-large-patch14-336`).
        revision: Commit sha, branch or tag of the checkpoint to read. A branch can move; a commit sha pins the
            embedding space.

    Example:
        ```{.python notest}
        from torch_pointcloud.layers import CLIPTextEncoder

        text_encoder = CLIPTextEncoder(
            "openai/clip-vit-large-patch14-336",
            revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        )
        text = text_encoder(["a chair", "a table"])  # (2, 768), unit norm
        ```
    """

    def __init__(self, name: str, revision: str = "main") -> None:
        super().__init__()
        self.name = name
        self.revision = revision
        self.tokenizer = CLIPTokenizer.from_pretrained(name, revision=revision)
        # The checkpoint also holds the vision tower, which `transformers` reports as unexpected keys at warning level.
        logger = logging.getLogger("transformers")
        level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            # `output_loading_info=True` returns `(model, loading_info)`, which the `transformers` annotation omits.
            loaded: Tuple[CLIPTextModelWithProjection, Dict[str, List[str]]] = (
                CLIPTextModelWithProjection.from_pretrained(  # type: ignore[assignment]
                    name,
                    revision=revision,
                    output_loading_info=True,
                )
            )
        finally:
            logger.setLevel(level)
        model, loading_info = loaded
        missing = sorted(loading_info["missing_keys"])
        if missing:
            raise ValueError(
                f"{name!r} is not a CLIP checkpoint: {len(missing)} weights of the text model and its projection are "
                f"missing from it (e.g. {missing[0]!r})."
            )

        self.model: nn.Module = model
        self.model.train(False)
        self.model.requires_grad_(False)
        projection_dim = model.config.projection_dim
        assert projection_dim is not None
        self._embed_dim = int(projection_dim)
        self.context_length = int(model.config.max_position_embeddings)

    @property
    def embed_dim(self) -> int:
        """Width $D$ of the projected text embeddings."""
        return self._embed_dim

    def forward(self, texts: Union[str, Sequence[str]]) -> Tensor:
        r"""Embed prompts into unit vectors.

        Args:
            texts: $K$ prompts, or one prompt as a string ($K = 1$). A prompt longer than the context length of the
                model (`context_length` tokens) is truncated.

        Returns:
            The unit-length text embeddings.

        Shape:
            - Output: $(K, D)$ where $D$ = `embed_dim`.
        """
        prompts = [texts] if isinstance(texts, str) else list(texts)
        if not prompts:
            raise ValueError("`texts` is empty: pass at least one prompt.")

        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        ).to(self.model.device)
        embeddings = self.model(**tokens).text_embeds
        return F.normalize(embeddings, dim=-1)
