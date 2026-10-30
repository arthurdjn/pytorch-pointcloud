import json
import string
from pathlib import Path
from typing import Dict
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.utils.imports import _TRANSFORMERS_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="transformers is not installed"),
]
transformers = pytest.importorskip("transformers")

from torch_pointcloud.layers.clip_text_encoder import CLIPTextEncoder  # noqa: E402


def _write_tokenizer(path: Path) -> int:
    letters = list(string.ascii_lowercase)
    tokens = [*letters, *(f"{letter}</w>" for letter in letters), "<|startoftext|>", "<|endoftext|>"]
    (path / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
    (path / "merges.txt").write_text("#version: 0.2\n")
    return len(tokens)


def _text_config(vocab_size: int) -> Dict[str, int]:
    return dict(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=77,
        projection_dim=16,
        bos_token_id=vocab_size - 2,
        eos_token_id=vocab_size - 1,
        pad_token_id=vocab_size - 1,
    )


@pytest.fixture(scope="module")
def clip_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly initialized two-layer CLIP and a letter-level tokenizer, so the tests need no download."""
    path = tmp_path_factory.mktemp("clip")
    vocab_size = _write_tokenizer(path)
    vision_config = dict(
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=30,
    )
    config = transformers.CLIPConfig(
        text_config=_text_config(vocab_size),
        vision_config=vision_config,
        projection_dim=16,
    )
    torch.manual_seed(0)
    transformers.CLIPModel(config).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def text_encoder(clip_dir: Path) -> CLIPTextEncoder:
    return CLIPTextEncoder(str(clip_dir))


def test_clip_text_encoder_forward(text_encoder: CLIPTextEncoder) -> None:
    prompts = ["a chair", "a table", "other"]
    with torch.no_grad():
        text = text_encoder(prompts)
    assert text.shape == (len(prompts), text_encoder.embed_dim)
    assert text_encoder.embed_dim == 16
    assert torch.allclose(text.norm(dim=1), torch.ones(len(prompts)), atol=1e-5)
    assert torch.isfinite(text).all()


def test_clip_text_encoder_matches_clip_text_projection(text_encoder: CLIPTextEncoder, clip_dir: Path) -> None:
    prompts = ["a chair", "a sofa"]
    clip = transformers.CLIPModel.from_pretrained(clip_dir).eval()
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
        first = text_encoder(["a chair"])
        second = text_encoder(["a chair"])
    assert torch.equal(first, second)


def test_clip_text_encoder_embeds_a_bare_string_as_one_prompt(text_encoder: CLIPTextEncoder) -> None:
    with torch.no_grad():
        text = text_encoder("a chair")
        expected = text_encoder(["a chair"])
    assert text.shape == (1, text_encoder.embed_dim)
    assert torch.equal(text, expected)


def test_clip_text_encoder_rejects_an_empty_vocabulary(text_encoder: CLIPTextEncoder) -> None:
    with pytest.raises(ValueError, match="`texts` is empty"):
        text_encoder([])


def test_clip_text_encoder_reads_the_given_revision(clip_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    tokenizer = Mock(wraps=transformers.CLIPTokenizer.from_pretrained)
    model = Mock(wraps=transformers.CLIPTextModelWithProjection.from_pretrained)
    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.CLIPTextModelWithProjection, "from_pretrained", model)
    text_encoder = CLIPTextEncoder(str(clip_dir), revision=revision)
    assert text_encoder.revision == revision
    assert tokenizer.call_args.kwargs["revision"] == revision
    assert model.call_args.kwargs["revision"] == revision


def test_clip_text_encoder_truncates_long_prompts(text_encoder: CLIPTextEncoder) -> None:
    prompt = " ".join(["chair"] * 100)  # 500 tokens, beyond the 77-token context
    with torch.no_grad():
        text = text_encoder([prompt])
    assert text.shape == (1, text_encoder.embed_dim)
    assert text_encoder.context_length == 77


def test_clip_text_encoder_rejects_a_checkpoint_without_the_projection(tmp_path: Path) -> None:
    vocab_size = _write_tokenizer(tmp_path)
    transformers.CLIPTextModel(transformers.CLIPTextConfig(**_text_config(vocab_size))).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="is not a CLIP checkpoint"):
        CLIPTextEncoder(str(tmp_path))
