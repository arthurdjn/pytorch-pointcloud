# KITTI

Synthetic KITTI object split: 3 frames of 1024 points with calibration, labels and blank images. Sweeps are ray-cast
against the annotated objects, so the boxes hold points.

```bash
uv run --no-sync python scripts/generate.py
```
