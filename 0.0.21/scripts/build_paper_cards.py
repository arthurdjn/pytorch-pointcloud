"""Render the preview assets behind the `paper()` documentation macro.

A page or a docstring asks for a card by calling the macro with an arXiv identifier:

    {{ paper("1612.00593") }}

This script scans the documentation and the sources for those calls and produces what
the macro needs at build time, so the macro itself never reaches the network:

- `docs/assets/papers/{key}-page.webp` is the top of page 1, cropped below the teaser
  figure. The bottom fade is baked into the alpha channel: a CSS `mask-image` or
  `mix-blend-mode` would do the same thing at the cost of a repaint on every scroll.
- Title and publication date come from the arXiv API and are cached in
  `docs/data/papers.json`.

Papers published outside arXiv are keyed by a slug listed in `SOURCES` below, which
carries what the API would otherwise answer:

    {{ paper("kitti-2012") }}

Rendering needs `pdftoppm` (poppler-utils) on the PATH.

Usage:
    uv run --no-sync python docs/scripts/build_paper_cards.py
    uv run --no-sync python docs/scripts/build_paper_cards.py --check
    uv run --no-sync python docs/scripts/build_paper_cards.py --force --refresh

A paper whose layout crops badly can say how much of the page to keep:

    {{ paper("1706.02413", crop=0.58) }}
"""

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "docs"
SRC_DIR = ROOT_DIR / "src"
PAPERS_DIR = DOCS_DIR / "assets" / "papers"
METADATA_PATH = DOCS_DIR / "data" / "papers.json"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "torch-pointcloud docs (https://github.com/arthurdjn/pytorch-pointcloud)"
ARXIV_DELAY = 3.0  # arXiv asks for one API request every three seconds
MAX_WORKERS = 4  # previews render concurrently; the API calls above stay serial

RENDER_DPI = 170
CROP_TOP = 0.055  # keep a slim page margin above the title
CROP_BOTTOM = 0.47  # cut below the teaser figure caption
FADE_START = 0.68
FADE_END = 0.97
WEBP_QUALITY = 86

MACRO = re.compile(
    r"""\{\{\s*paper\(\s*["'](?P<key>[\w./-]+)["'](?:\s*,\s*crop\s*=\s*(?P<crop>[0-9.]+))?\s*,?\s*\)\s*\}\}"""
)

SOURCES: Dict[str, Dict[str, str]] = {
    "kitti-2012": {
        "title": "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite",
        "date": "June 2012",
        "label": "CVPR",
        "icon": "cvf",
        "url": "https://www.cvlibs.net/publications/Geiger2012CVPR.pdf",
        "pdf": "https://www.cvlibs.net/publications/Geiger2012CVPR.pdf",
    },
    "s3dis-2016": {
        "title": "3D Semantic Parsing of Large-Scale Indoor Spaces",
        "date": "June 2016",
        "label": "CVPR",
        "icon": "cvf",
        "url": "https://openaccess.thecvf.com/content_cvpr_2016/html/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.html",
        "pdf": "https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf",
    },
    "second-2018": {
        "title": "SECOND: Sparsely Embedded Convolutional Detection",
        "date": "October 2018",
        "label": "Sensors",
        "url": "https://www.mdpi.com/1424-8220/18/10/3337",
        "pdf": "https://mdpi-res.com/d_attachment/sensors/sensors-18-03337/article_deploy/sensors-18-03337.pdf",
    },
    "shapenetpart-2016": {
        "title": "A Scalable Active Framework for Region Annotation in 3D Shape Collections",
        "date": "November 2016",
        "label": "ACM TOG",
        "icon": "simple-acm",
        "url": "https://dl.acm.org/doi/10.1145/2980179.2980238",
        "pdf": "https://cs.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf",
    },
    "sunrgbd-2015": {
        "title": "SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite",
        "date": "June 2015",
        "label": "CVPR",
        "icon": "cvf",
        "url": "https://openaccess.thecvf.com/content_cvpr_2015/html/Song_SUN_RGB-D_A_2015_CVPR_paper.html",
        "pdf": "https://openaccess.thecvf.com/content_cvpr_2015/papers/Song_SUN_RGB-D_A_2015_CVPR_paper.pdf",
    },
}


def call_sites() -> Dict[str, float]:
    """Collect every paper the documentation and the sources ask for.

    Returns:
        Mapping of paper key to the fraction of the first page to keep.

    Example:
        >>> sorted(call_sites())  # doctest: +SKIP
        ['1612.00593', '1706.02413', 'kitti-2012']
    """
    wanted: Dict[str, float] = {}
    for path in [*sorted(DOCS_DIR.rglob("*.md")), *sorted(SRC_DIR.rglob("*.py"))]:
        for match in MACRO.finditer(path.read_text()):
            wanted[match["key"]] = float(match["crop"]) if match["crop"] else CROP_BOTTOM
    return wanted


def fetch_metadata(arxiv_id: str) -> Dict[str, object]:
    """Read the title and publication date for one paper from the arXiv API.

    Args:
        arxiv_id: Bare arXiv identifier, for example `1612.00593`.

    Returns:
        Mapping with `title` and `date` keys.

    Example:
        >>> fetch_metadata("1612.00593")["title"]  # doctest: +SKIP
        'PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation'
    """
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        entry = ET.fromstring(response.read()).find("atom:entry", ATOM_NS)
    if entry is None:
        raise RuntimeError(f"arXiv returned no entry for {arxiv_id}")

    def text(element: ET.Element) -> str:
        return re.sub(r"\s+", " ", element.text or "").strip()

    title = entry.find("atom:title", ATOM_NS)
    published = entry.find("atom:published", ATOM_NS)
    if title is None or published is None:
        raise RuntimeError(f"arXiv entry for {arxiv_id} is missing a title or date")
    date = datetime.strptime(text(published), "%Y-%m-%dT%H:%M:%SZ")
    return {"title": text(title), "date": date.strftime("%B %Y")}


def render_preview(pdf_url: str, crop: float) -> bytes:
    """Render the top of page 1 as a WebP whose bottom fade lives in the alpha channel.

    Args:
        pdf_url: Direct link to the paper's PDF.
        crop: Fraction of the page height to keep. Layouts differ, so a paper whose
            title block or teaser figure falls badly can override it at the call site.

    Returns:
        Encoded WebP bytes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        pdf = work / "paper.pdf"
        request = urllib.request.Request(pdf_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            pdf.write_bytes(response.read())
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                str(RENDER_DPI),
                "-f",
                "1",
                "-l",
                "1",
                "-singlefile",
                str(pdf),
                str(work / "page1"),
            ],
            check=True,
            capture_output=True,
        )
        page = Image.open(work / "page1.png").convert("RGB")

    width, height = page.size
    page = page.crop((0, round(CROP_TOP * height), width, round(crop * height))).convert("RGBA")
    page.putalpha(fade(page.size))

    buffer = io.BytesIO()
    page.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=6)
    return buffer.getvalue()


def fade(size: Tuple[int, int]) -> Image.Image:
    """Build the alpha ramp that dissolves the bottom of the page into the card.

    Args:
        size: Target `(width, height)` in pixels.

    Returns:
        Single channel mask, opaque at the top and transparent at the bottom.
    """
    width, height = size
    start, end = FADE_START * height, FADE_END * height
    column = Image.new("L", (1, height))
    for y in range(height):
        ratio = (y - start) / (end - start)
        column.putpixel((0, y), 255 if ratio <= 0 else 0 if ratio >= 1 else round(255 * (1 - ratio)))
    return column.resize((width, height))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report missing and stale assets without writing")
    parser.add_argument("--force", action="store_true", help="re-render preview assets that already exist")
    parser.add_argument("--refresh", action="store_true", help="refetch metadata instead of reading the cache")
    args = parser.parse_args()

    metadata: Dict[str, Dict[str, object]] = {}
    if METADATA_PATH.exists() and not args.refresh:
        metadata = json.loads(METADATA_PATH.read_text())

    wanted = call_sites()
    if not wanted:
        print("no paper() call sites found")
        return 0

    # `SOURCES` is the record for a paper published outside arXiv, so it overrides the
    # cache rather than seeding it, and an edit there lands without a refetch.
    for key in wanted.keys() & SOURCES.keys():
        crop = metadata.get(key, {}).get("crop")
        metadata[key] = {field: value for field, value in SOURCES[key].items() if field != "pdf"}
        if crop is not None:
            metadata[key]["crop"] = crop

    stale: List[str] = []
    failed: List[str] = []
    todo: List[Tuple[str, str, float]] = []
    for key, crop in sorted(wanted.items()):
        asset = PAPERS_DIR / f"{key}-page.webp"
        needs_meta = key not in metadata
        needs_asset = args.force or not asset.exists() or metadata.get(key, {}).get("crop") != crop
        if not (needs_meta or needs_asset):
            continue
        if args.check:
            stale.append(f"stale: {asset.relative_to(ROOT_DIR)}")
            continue
        if needs_asset:
            todo.append((key, SOURCES[key]["pdf"] if key in SOURCES else f"https://arxiv.org/pdf/{key}", crop))
        if not needs_meta:
            continue
        try:
            metadata[key] = fetch_metadata(key)
        except Exception as error:  # one unreachable paper must not abandon the rest of the run
            failed.append(f"failed: {key} ({error})")
        time.sleep(ARXIV_DELAY)  # the arXiv API is queried one paper at a time, as they ask

    if todo:
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        renders = {pool.submit(render_preview, pdf, crop): (key, crop) for key, pdf, crop in todo}
        for future in as_completed(renders):
            key, crop = renders[future]
            asset = PAPERS_DIR / f"{key}-page.webp"
            try:
                asset.write_bytes(future.result())
            except Exception as error:
                failed.append(f"failed: {key} ({error})")
                continue
            metadata.setdefault(key, {})["crop"] = crop
            print(f"rendered: {asset.relative_to(ROOT_DIR)} ({asset.stat().st_size // 1024} KB)")

    for asset in sorted(PAPERS_DIR.glob("*-page.webp")):
        if asset.name[: -len("-page.webp")] in wanted:
            continue
        stale.append(f"orphan: {asset.relative_to(ROOT_DIR)}")
        if not args.check:
            asset.unlink()

    if not args.check:
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cached = {k: metadata[k] for k in sorted(wanted) if k in metadata}
        METADATA_PATH.write_text(json.dumps(cached, indent=2, sort_keys=True) + "\n")

    for line in [*stale, *failed]:
        print(line)
    print(f"{len(wanted)} paper(s), {len(stale)} stale, {len(failed)} failed")
    return 1 if failed or (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
