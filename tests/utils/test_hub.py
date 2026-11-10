from pathlib import Path

import pytest

from torch_pointcloud.utils.hub import load_hub_token


def test_load_hub_token_prefers_the_environment_over_the_login_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", tmp_path.as_posix())
    assert load_hub_token() is None

    (tmp_path / "huggingface").mkdir()
    (tmp_path / "huggingface" / "token").write_text("hf_from_file\n")
    assert load_hub_token() == "hf_from_file"

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert load_hub_token() == "hf_from_env"
