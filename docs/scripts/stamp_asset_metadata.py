"""Embed author, project and license metadata into the committed documentation assets.

Rendered figures, sample clouds and the project mark carry no attribution once they leave the
repository, so the credit is written into the files themselves. The script walks a set of roots and
stamps every file whose format it knows:

- PNG: `tEXt` chunks (`Title`, `Author`, `Copyright`, ...) plus an `XML:com.adobe.xmp` packet, which
  is what image tooling actually reads. Chunks are spliced around the existing ones, so the pixel
  data is never re-encoded.
- WebP: an `XMP ` RIFF chunk, promoting the file to the extended (`VP8X`) layout when it is still in
  the simple one. The image chunks are copied through untouched.
- PLY: `comment` lines in the ASCII header, ahead of `end_header`.
- SVG: a Dublin Core `<metadata>` block.

Anything else is reported as having no handler rather than skipped silently, so a newly added asset
format is visible instead of quietly going out unattributed.

Assets whose content comes from elsewhere are covered by `RULES`: a rule either credits the source
alongside the project (the sample clouds are exported from public benchmarks) or leaves the file
alone entirely (third-party brand icons, paper page crops).

Re-running replaces the managed metadata instead of appending to it, and files that already carry
the exact metadata are left untouched.

Usage:
    uv run --no-sync python docs/scripts/stamp_asset_metadata.py
    uv run --no-sync python docs/scripts/stamp_asset_metadata.py --check
    uv run --no-sync python docs/scripts/stamp_asset_metadata.py --list
    uv run --no-sync python docs/scripts/stamp_asset_metadata.py docs/assets/transforms --ext .png
    uv run --no-sync python docs/scripts/stamp_asset_metadata.py --exclude papers --exclude '*_scene.*'

Patterns are `fnmatch` globs tested against the repository-relative path and against the file or
directory name, so `--exclude papers`, `--exclude 'assets/papers/*'` and `--exclude '*.webp'` all
select the same files here. An excluded directory is pruned rather than walked.
"""

import argparse
import fnmatch
import os
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

AUTHOR = "Arthur Dujardin"
PROJECT = "torch-pointcloud"
PROJECT_URL = "https://github.com/arthurdjn/pytorch-pointcloud"
COPYRIGHT = f"Copyright (c) 2024-2026 {AUTHOR}"
LICENSE_NAME = "Apache-2.0"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"

REPO_DIR = Path(__file__).resolve().parents[2]

DEFAULT_ROOTS = ("docs/assets", "docs/overrides/.icons")
DEFAULT_EXCLUDES = ("__pycache__", ".git")


@dataclass(frozen=True)
class Rule:
    """How one family of paths is treated. The first matching rule wins."""

    pattern: str
    source: Optional[str] = None  # external origin, credited alongside the project
    skip: Optional[str] = None  # why the file is left untouched


RULES = (
    Rule("*/.icons/pytorch-pointcloud*.svg"),
    Rule("*/.icons/*.svg", skip="third-party brand icon"),
    Rule("*/assets/papers/*", skip="page crop of a third-party paper"),
    Rule("*/assets/data/sample.ply", source="ShapeNetPart"),
    Rule("*/assets/data/sample_mesh.ply", source="ModelNet"),
    Rule("*/assets/data/sample_scene*.ply", source="ScanNet"),
    Rule("*/assets/data/sample_lidar_*.ply", source="SemanticKITTI"),
)

SUMMARIES = (
    ("*/assets/pytorch-pointcloud.*", f"Project logo of {PROJECT}."),
    ("*/assets/transforms/*", f"Transform gallery figure for the {PROJECT} documentation."),
    ("*/assets/data/*", f"Point cloud sample the {PROJECT} documentation figures are rendered from."),
    ("*/.icons/*", f"Project mark of {PROJECT}."),
    ("*", f"Documentation asset of {PROJECT}."),
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
XMP_KEYWORD = "XML:com.adobe.xmp"
RIFF_SIGNATURE = b"RIFF"
WEBP_FORM = b"WEBP"
XMP_CHUNK = b"XMP "
VP8X_ALPHA_FLAG = 0x10
VP8X_XMP_FLAG = 0x04
PLY_COMMENT_TAG = f"comment {PROJECT}:"


@dataclass(frozen=True)
class Asset:
    """One file to stamp, and the metadata it carries."""

    path: Path
    rel: str
    title: str
    summary: str
    source: Optional[str] = None

    @property
    def description(self) -> str:
        if self.source is None:
            return self.summary
        return f"{self.summary} Derived from {self.source}; the source terms apply."


def relative(path: Path) -> str:
    """Repository-relative POSIX path, or an absolute one for a path outside the repository."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_DIR):
        return resolved.relative_to(REPO_DIR).as_posix()
    return resolved.as_posix()


def matches(rel: str, pattern: str) -> bool:
    """Test a glob against a path, its parent directories, and the name of each."""
    path = PurePosixPath(rel)
    for candidate in (path, *path.parents):
        text = candidate.as_posix()
        if text in (".", "/"):
            continue
        if fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(text, f"*/{pattern}"):
            return True
        if fnmatch.fnmatch(candidate.name, pattern):
            return True
    return False


def rule_for(rel: str) -> Rule:
    return next((rule for rule in RULES if matches(rel, rule.pattern)), Rule(rel))


def describe(path: Path) -> Asset:
    """Build the metadata for one file from its path."""
    rel = relative(path)
    summary = next(text for pattern, text in SUMMARIES if matches(rel, pattern))
    return Asset(
        path=path,
        rel=rel,
        title=f"{path.stem.replace('_', ' ')} - {PROJECT}",
        summary=summary,
        source=rule_for(rel).source,
    )


def xmp_packet(asset: Asset) -> str:
    r"""Build an XMP packet carrying `dc:creator`, `dc:rights` and the license statement."""
    return f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/"
    xmlns:cc="http://creativecommons.org/ns#">
   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{asset.title}</rdf:li></rdf:Alt></dc:title>
   <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{asset.description}</rdf:li></rdf:Alt></dc:description>
   <dc:creator><rdf:Seq><rdf:li>{AUTHOR}</rdf:li></rdf:Seq></dc:creator>
   <dc:rights><rdf:Alt><rdf:li xml:lang="x-default">{COPYRIGHT}</rdf:li></rdf:Alt></dc:rights>
   <dc:source>{PROJECT_URL}</dc:source>
   <xmpRights:Marked>True</xmpRights:Marked>
   <xmpRights:WebStatement rdf:resource="{PROJECT_URL}"/>
   <cc:license rdf:resource="{LICENSE_URL}"/>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "big") + kind + payload + zlib.crc32(kind + payload).to_bytes(4, "big")


def read_png_chunks(blob: bytes) -> List[Tuple[bytes, bytes]]:
    if not blob.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    chunks: List[Tuple[bytes, bytes]] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(blob):
        size = int.from_bytes(blob[offset : offset + 4], "big")
        kind = blob[offset + 4 : offset + 8]
        chunks.append((kind, blob[offset + 8 : offset + 8 + size]))
        offset += 12 + size
    return chunks


def stamp_png(blob: bytes, asset: Asset) -> bytes:
    entries = {
        "Title": asset.title,
        "Author": AUTHOR,
        "Description": asset.description,
        "Copyright": COPYRIGHT,
        "Source": PROJECT_URL,
        "Disclaimer": f"Licensed under the {LICENSE_NAME} License, {LICENSE_URL}",
    }
    managed = set(entries)
    kept = [
        (kind, payload)
        for kind, payload in read_png_chunks(blob)
        if not (kind == b"tEXt" and payload.split(b"\x00", 1)[0].decode("latin-1") in managed)
        and not (kind == b"iTXt" and payload.split(b"\x00", 1)[0].decode("latin-1") == XMP_KEYWORD)
    ]
    text = [png_chunk(b"tEXt", f"{k}\x00{v}".encode("latin-1")) for k, v in entries.items()]
    # iTXt payload: keyword, compression flag, compression method, language tag, translated keyword, text.
    xmp = xmp_packet(asset).encode("utf-8")
    text.append(png_chunk(b"iTXt", XMP_KEYWORD.encode("latin-1") + b"\x00\x00\x00\x00\x00" + xmp))
    # Text chunks are spliced between IHDR and everything that follows it.
    head = png_chunk(*kept[0])
    rest = b"".join(png_chunk(kind, payload) for kind, payload in kept[1:])
    return PNG_SIGNATURE + head + b"".join(text) + rest


def riff_chunk(kind: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) % 2 else b""
    return kind + len(payload).to_bytes(4, "little") + payload + pad


def read_riff_chunks(blob: bytes) -> List[Tuple[bytes, bytes]]:
    if not (blob.startswith(RIFF_SIGNATURE) and blob[8:12] == WEBP_FORM):
        raise ValueError("not a WebP file")
    chunks: List[Tuple[bytes, bytes]] = []
    offset = 12
    end = min(len(blob), 8 + int.from_bytes(blob[4:8], "little"))
    while offset + 8 <= end:
        kind = blob[offset : offset + 4]
        size = int.from_bytes(blob[offset + 4 : offset + 8], "little")
        chunks.append((kind, blob[offset + 8 : offset + 8 + size]))
        offset += 8 + size + size % 2
    return chunks


def webp_canvas(kind: bytes, payload: bytes) -> Tuple[int, int, bool]:
    """Canvas width, height and alpha flag of a simple-format WebP image chunk."""
    if kind == b"VP8 ":
        # Key frame: a 3-byte frame tag, the start code, then 14-bit width and height.
        if payload[3:6] != b"\x9d\x01\x2a":
            raise ValueError("unsupported VP8 frame")
        width = int.from_bytes(payload[6:8], "little") & 0x3FFF
        height = int.from_bytes(payload[8:10], "little") & 0x3FFF
        return width, height, False
    if kind == b"VP8L":
        if payload[0] != 0x2F:
            raise ValueError("unsupported VP8L stream")
        bits = int.from_bytes(payload[1:5], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, bool((bits >> 28) & 1)
    raise ValueError(f"unsupported WebP image chunk {kind.decode('latin-1')}")


def stamp_webp(blob: bytes, asset: Asset) -> bytes:
    chunks = [(kind, payload) for kind, payload in read_riff_chunks(blob) if kind != XMP_CHUNK]
    if not chunks:
        raise ValueError("WebP file carries no chunk")
    if chunks[0][0] == b"VP8X":
        flags = bytearray(chunks[0][1])
        flags[0] |= VP8X_XMP_FLAG
        chunks[0] = (b"VP8X", bytes(flags))
    else:
        width, height, alpha = webp_canvas(*chunks[0])
        header = bytes([VP8X_XMP_FLAG | (VP8X_ALPHA_FLAG if alpha else 0), 0, 0, 0])
        chunks.insert(0, (b"VP8X", header + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")))
    chunks.append((XMP_CHUNK, xmp_packet(asset).encode("utf-8")))
    body = WEBP_FORM + b"".join(riff_chunk(kind, payload) for kind, payload in chunks)
    return RIFF_SIGNATURE + len(body).to_bytes(4, "little") + body


def stamp_ply(blob: bytes, asset: Asset) -> bytes:
    end = blob.index(b"end_header")
    header, payload = blob[:end].decode("ascii"), blob[end:]
    newline = "\r\n" if "\r\n" in header else "\n"
    lines = [line for line in header.splitlines() if not line.startswith(PLY_COMMENT_TAG)]
    comments = [
        f"{PLY_COMMENT_TAG} {asset.title}",
        f"{PLY_COMMENT_TAG} {asset.summary}",
        f"{PLY_COMMENT_TAG} prepared by {AUTHOR}, {COPYRIGHT}, {LICENSE_NAME}, {PROJECT_URL}",
    ]
    if asset.source is not None:
        comments.append(f"{PLY_COMMENT_TAG} geometry derived from {asset.source}; the source terms apply")
    # `format` is mandatory on the second line; comments follow it.
    at = next(i for i, line in enumerate(lines) if line.startswith("format")) + 1
    stamped = lines[:at] + comments + lines[at:]
    return (newline.join(stamped) + newline).encode("ascii") + payload


def stamp_svg(blob: bytes, asset: Asset) -> bytes:
    text = re.sub(r"\n?\s*<metadata>.*?</metadata>", "", blob.decode("utf-8"), flags=re.DOTALL)
    metadata = f"""
    <metadata>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:dc="http://purl.org/dc/elements/1.1/"
                 xmlns:cc="http://creativecommons.org/ns#">
            <cc:Work rdf:about="">
                <dc:title>{asset.title}</dc:title>
                <dc:description>{asset.description}</dc:description>
                <dc:creator><cc:Agent><dc:title>{AUTHOR}</dc:title></cc:Agent></dc:creator>
                <dc:rights><cc:Agent><dc:title>{COPYRIGHT}</dc:title></cc:Agent></dc:rights>
                <dc:source>{PROJECT_URL}</dc:source>
                <cc:license rdf:resource="{LICENSE_URL}"/>
            </cc:Work>
        </rdf:RDF>
    </metadata>"""
    opening = re.search(r"<svg\b[^>]*>", text)
    if opening is None:
        raise ValueError("no <svg> element")
    return (text[: opening.end()] + metadata + text[opening.end() :]).encode("utf-8")


HANDLERS: Dict[str, Callable[[bytes, Asset], bytes]] = {
    ".png": stamp_png,
    ".webp": stamp_webp,
    ".ply": stamp_ply,
    ".svg": stamp_svg,
}


def selected(rel: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    if any(matches(rel, pattern) for pattern in excludes):
        return False
    return not includes or any(matches(rel, pattern) for pattern in includes)


def walk(
    roots: Sequence[Path], includes: Sequence[str], excludes: Sequence[str], extensions: Sequence[str]
) -> Iterator[Path]:
    """Yield every file under `roots` that survives the include, exclude and extension filters."""
    for root in roots:
        if root.is_file():
            candidates: Iterator[Path] = iter([root])
        else:
            candidates = walk_tree(root, excludes)
        for path in candidates:
            if extensions and path.suffix.lower() not in extensions:
                continue
            if selected(relative(path), includes, excludes):
                yield path


def walk_tree(root: Path, excludes: Sequence[str]) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if selected(relative(Path(dirpath) / d), (), excludes))
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help=f"Files or directories to walk (default: {', '.join(DEFAULT_ROOTS)}).")
    parser.add_argument("--check", action="store_true", help="Report unstamped assets without writing.")
    parser.add_argument("--list", action="store_true", help="Report what each file would get, without writing.")
    parser.add_argument(
        "--ext", action="append", default=[], metavar="EXT", help="Only stamp this extension, repeatable."
    )
    parser.add_argument(
        "--include", action="append", default=[], metavar="GLOB", help="Only stamp matching paths, repeatable."
    )
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB", help="Skip matching paths, repeatable."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    roots = [Path(p) if Path(p).is_absolute() else REPO_DIR / p for p in (args.paths or DEFAULT_ROOTS)]
    missing = [root for root in roots if not root.exists()]
    if missing:
        for root in missing:
            print(f"no such path: {relative(root)}", file=sys.stderr)
        return 2

    extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.ext]
    excludes = list(DEFAULT_EXCLUDES) + args.exclude

    stale: List[Asset] = []
    fresh = 0
    skipped: List[Tuple[str, str]] = []
    unhandled: List[str] = []
    failed: List[Tuple[str, str]] = []
    for path in walk(roots, args.include, excludes, extensions):
        rel = relative(path)
        rule = rule_for(rel)
        if rule.skip is not None:
            skipped.append((rel, rule.skip))
            continue
        handler = HANDLERS.get(path.suffix.lower())
        if handler is None:
            unhandled.append(rel)
            continue
        asset = describe(path)
        if args.list:
            print(f"{rel}: {asset.title} | {asset.description}")
            continue
        blob = path.read_bytes()
        try:
            stamped = handler(blob, asset)
        except (OSError, ValueError, IndexError) as error:
            failed.append((rel, str(error)))
            continue
        if stamped == blob:
            fresh += 1
            continue
        stale.append(asset)
        if not args.check:
            path.write_bytes(stamped)

    for rel, reason in skipped:
        print(f"skipped: {rel} ({reason})")
    for rel in unhandled:
        print(f"no handler: {rel}")
    for rel, reason in failed:
        print(f"failed: {rel} ({reason})", file=sys.stderr)
    if args.list:
        return 0

    label = "unstamped" if args.check else "stamped"
    for asset in stale:
        print(f"{label}: {asset.rel}")
    print(f"{len(stale)} {label}, {fresh} already stamped, {len(skipped)} skipped, {len(unhandled)} without a handler")
    return 1 if failed or (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
