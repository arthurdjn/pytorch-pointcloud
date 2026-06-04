#!/usr/bin/env bash
set -euo pipefail

# Build the tiny subset zips from the real release, then process them into the
# subsampled processed/ cache. Reads from $TORCH_POINTCLOUD_DATA_DIR/SunRGBD.
uv run --no-sync python scripts/generate.py raw
uv run --no-sync python scripts/generate.py process
