"""Hugging Face Hub file URLs and the token of the environment."""

import os
import re
from pathlib import Path
from typing import Optional

HF_ENDPOINT = "https://huggingface.co"
HF_URL_RE = re.compile(r"^hf://(?P<namespace>[^/]+)/(?P<repo>[^/]+)/resolve/(?P<revision>[^/]+)/(?P<file>.+)$")


def load_hub_token() -> Optional[str]:
    """Return the Hugging Face Hub token of the environment: `HF_TOKEN`, else the file `hf auth login` wrote
    (`HF_TOKEN_PATH`, by default `~/.cache/huggingface/token`). The weights of a gated repository download with it.

    Returns:
        The token, or `None` when none is set.
    """
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token

    cache_home = os.getenv("XDG_CACHE_HOME", Path(Path.home(), ".cache").as_posix())
    hf_home = os.getenv("HF_HOME", Path(cache_home, "huggingface").as_posix())
    path = Path(os.getenv("HF_TOKEN_PATH", Path(hf_home, "token").as_posix()))
    if not path.is_file():
        return None
    return path.read_text().strip() or None
