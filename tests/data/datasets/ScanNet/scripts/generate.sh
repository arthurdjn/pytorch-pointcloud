#!/usr/bin/env bash
set -euo pipefail

# Subsample real scenes for both v1 and v2; raw v1 reuses v2 PLY/aggregation files
# but ships under a v1-labelled labels file (matching the real version split).
for VER in v1 v2; do
    for SPLIT in train val test; do
        uv run --no-sync python scripts/generate.py raw ./raw --version "$VER" --split "$SPLIT" --ignore-warnings
    done
done

for VER in v1 v2; do
    for SPLIT in train val test; do
        uv run --no-sync python scripts/generate.py process ./raw --version "$VER" --split "$SPLIT" --ignore-warnings
    done
done
