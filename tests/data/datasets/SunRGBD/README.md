# SUN RGB-D

Synthetic SUN RGB-D: the two release zips with 3 scenes per split (depth, RGB, calibration, boxes), plus the
`processed/` cache subsampled to 2048 points. The last scene of each split has no box.

```bash
uv run --no-sync python scripts/generate.py raw
uv run --no-sync python scripts/generate.py process
```
