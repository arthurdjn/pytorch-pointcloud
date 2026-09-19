#!/usr/bin/env bash
set -euo pipefail

# Generate the synthetic scenes for both v1 and v2; raw v1 holds the same scenes as v2
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
