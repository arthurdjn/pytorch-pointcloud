#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/generate.py raw ./raw
