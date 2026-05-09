#!/usr/bin/env bash
set -euo pipefail

uv run --no-sync python scripts/generate.py raw ./raw
