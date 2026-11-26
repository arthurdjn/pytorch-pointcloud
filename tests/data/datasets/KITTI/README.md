# KITTI

Synthetic KITTI object split: 3 frames of 1024 points with calibration, labels and blank images. Sweeps are ray-cast
against the annotated objects, so the boxes hold points. `ImageSets` lists two frames in `train.txt` and one in
`val.txt`.

```bash
uv run --no-sync python scripts/generate.py
```
