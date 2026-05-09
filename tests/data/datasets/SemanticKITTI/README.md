# SemanticKITTI

Tiny SemanticKITTI fixture obtained by subsampling real scans from the original
release (1 sequence per split, 2 frames each, 1024 points per scan).

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/SemanticKITTI/raw/`
by default (override with `--src-dir`).

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```

This yields the same on-disk layout as the real dataset:

```
raw/sequences/{seq}/velodyne/{frame:06d}.bin    # float32 (N, 4) = (x, y, z, intensity)
raw/sequences/{seq}/labels/{frame:06d}.label    # uint32  (N,)   = (instance << 16) | semantic_id
```

The test split (sequence 11 in the real release) is shipped without `labels/`,
matching the held-out test labels of the official benchmark.
