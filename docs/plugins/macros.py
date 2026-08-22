"""Macros callable from the documentation Markdown and from the source docstrings.

Zensical loads this module through its macros extension, configured in `zensical.toml`.
Every Markdown page and every docstring rendered by mkdocstrings passes through Jinja2
first, so a macro registered here is available in both:

```markdown
{{ paper("1706.02413") }}
```

Assets and metadata are produced ahead of the build by `docs/scripts/build_paper_cards.py`
(`make papers`), which scans the same call sites. A macro never reaches the network.
"""

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import jinja2

DOCS_DIR = Path(__file__).resolve().parents[1]
PAPERS_DIR = DOCS_DIR / "assets" / "papers"
METADATA_PATH = DOCS_DIR / "data" / "papers.json"

CARD = """<div class="paper-card paper-card--page" markdown="1">
{preview}
<p class="paper-card__meta{modifier}" markdown="span">{cite}{date}</p>

</div>"""

PREVIEW = """
<img class="paper-card__page" src="{src}" alt="First page of {title}">
"""

ARXIV_CITE = ":arxiv-name: [`{key}`](https://arxiv.org/abs/{key})"
WORDMARK = " paper-card__meta--wordmark"


def alt_text(title: str) -> str:
    """Flatten a paper title into prose that survives an HTML attribute.

    Arithmatex scans the whole text stream, so a title carrying inline math (`PointCNN:
    Convolution On $\\mathcal{X}$-Transformed Points`) would have its `$...$` lifted out
    of the `alt` attribute, splitting the tag and spilling the rest onto the page.

    Args:
        title: Paper title as the arXiv API reports it.

    Returns:
        The title with TeX commands, math delimiters and quotes removed.

    Example:
        >>> alt_text(r"Convolution On $\\mathcal{X}$-Transformed Points")
        'Convolution On X-Transformed Points'
    """
    text = re.sub(r"\\[a-zA-Z]+", "", title)
    return re.sub(r'[${}"]', "", text)


def define_env(env: Any) -> None:
    """Register the documentation macros with the Jinja2 environment.

    Args:
        env: Macro environment supplied by the zensical macros extension.
    """
    env.macro(paper)


@jinja2.pass_context
def paper(context: jinja2.runtime.Context, key: str, crop: Optional[float] = None) -> str:
    r"""Render a preview card for a paper, linking to where it was published.

    The card shows the top of page 1 and the paper's identifier and date. It degrades to
    a plain identifier and link when the preview has not been rendered yet.

    Args:
        key: Bare arXiv identifier, for example `1706.02413`, or the slug of a paper
            published elsewhere, for example `kitti-2012`.
        crop: Fraction of the first page to keep. Read by `build_paper_cards.py`, which
            renders the asset, and ignored here.

    Returns:
        Markdown, re-parsed by the surrounding page.

    Example:
        ```markdown
        {{ paper("1706.02413", crop=0.58) }}
        ```
    """
    del crop
    metadata: Dict[str, Dict[str, str]] = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())

    meta = metadata.get(key, {})
    asset = PAPERS_DIR / f"{key}-page.webp"
    preview = ""
    if meta and asset.exists():
        page = context.get("page")
        depth = len(PurePosixPath(page.path).parts) - 1 if page else 0
        src = "../" * depth + asset.relative_to(DOCS_DIR).as_posix()
        preview = PREVIEW.format(src=src, title=alt_text(meta["title"]))

    if "url" not in meta:
        cite, modifier = ARXIV_CITE.format(key=key), WORDMARK
    else:
        icon = f":{meta['icon']}: " if "icon" in meta else ""
        cite, modifier = f"{icon}[{meta['label']}]({meta['url']})", ""

    return CARD.format(
        preview=preview,
        cite=cite,
        modifier=modifier,
        date=f" &middot; {meta['date']}" if meta else "",
    )
