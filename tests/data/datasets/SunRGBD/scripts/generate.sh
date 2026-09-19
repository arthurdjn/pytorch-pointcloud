#!/usr/bin/env bash
set -euo pipefail

# Build the tiny synthetic zips, then process them into the subsampled processed/ cache.
uv run --no-sync python scripts/generate.py raw
uv run --no-sync python scripts/generate.py process
