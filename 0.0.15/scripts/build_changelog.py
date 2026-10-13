"""Render `CHANGELOG.md` as the changelog page of the docs (`docs/changelog.md`).

The page is the file as is, minus the `## Unreleased` section: a development build keeps it with `--unreleased`, a
release build drops it.

Usage:
    uv run --no-sync python docs/scripts/build_changelog.py [--unreleased]
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "CHANGELOG.md"
TARGET = ROOT / "docs" / "changelog.md"

UNRELEASED = re.compile(r"^## Unreleased\n.*?(?=^## )", re.DOTALL | re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CHANGELOG.md as the changelog page of the docs.")
    parser.add_argument("--unreleased", action="store_true", help="Keep the `## Unreleased` section.")
    args = parser.parse_args()

    text = SOURCE.read_text()
    if not args.unreleased:
        text = UNRELEASED.sub("", text)
    TARGET.write_text(text)
    print(f"wrote {TARGET.relative_to(ROOT)}, {'with' if args.unreleased else 'without'} the unreleased section")


if __name__ == "__main__":
    main()
