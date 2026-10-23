import pytest
import torch

from torch_pointcloud.utils.imports import _TRANSFORMERS_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="transformers is not installed"),
]
transformers = pytest.importorskip("transformers")

from torch_pointcloud.layers.clip_text_encoder import CLIPTextEncoder  # noqa: E402

CLIP_NAME = "openai/clip-vit-base-patch32"


@pytest.fixture(scope="module")
def text_encoder() -> CLIPTextEncoder:
    return CLIPTextEncoder(CLIP_NAME)


def test_clip_text_encoder_forward(text_encoder: CLIPTextEncoder) -> None:
    prompts = ["a chair in a scene", "a table in a scene", "other"]
    with torch.no_grad():
        text = text_encoder(prompts)
    assert text.shape == (len(prompts), text_encoder.embed_dim)
    assert text_encoder.embed_dim == 512
    assert torch.allclose(text.norm(dim=1), torch.ones(len(prompts)), atol=1e-5)
    assert torch.isfinite(text).all()


def test_clip_text_encoder_matches_clip_text_projection(text_encoder: CLIPTextEncoder) -> None:
    prompts = ["a chair in a scene", "a sofa in a scene"]
    clip = transformers.CLIPModel.from_pretrained(CLIP_NAME).eval()
    tokens = text_encoder.tokenizer(prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        text = text_encoder(prompts)
        projected = clip.text_projection(clip.text_model(**tokens).pooler_output)
        expected = torch.nn.functional.normalize(projected, dim=-1)
    assert torch.allclose(text, expected, atol=1e-6)


def test_clip_text_encoder_is_frozen(text_encoder: CLIPTextEncoder) -> None:
    assert not text_encoder.model.training
    assert all(not p.requires_grad for p in text_encoder.parameters())
    with torch.no_grad():
        first = text_encoder(["a chair in a scene"])
        second = text_encoder(["a chair in a scene"])
    assert torch.equal(first, second)


def test_clip_text_encoder_truncates_long_prompts(text_encoder: CLIPTextEncoder) -> None:
    prompt = " ".join(["chair"] * 100)  # beyond the 77-token context of CLIP
    with torch.no_grad():
        text = text_encoder([prompt])
    assert text.shape == (1, text_encoder.embed_dim)
