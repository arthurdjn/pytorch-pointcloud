#!/usr/bin/env bash
set -euo pipefail

uv run --no-sync python scripts/generate.py raw ./raw --variant 10
uv run --no-sync python scripts/generate.py process ./raw --variant 10
