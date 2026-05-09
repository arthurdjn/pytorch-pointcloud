# Paris-Lille-3D

Tiny Paris-Lille-3D fixture obtained by subsampling the four real PLY scans from
the 10-class benchmark (`Lille1_1`, `Lille1_2`, `Lille2`, `Paris`). Each file
ships with 1024 randomly sampled vertices keeping the original schema
(`x`, `y`, `z`, `reflectance`, `class`).

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/ParisLille3D/raw/`
by default (override with `--src-dir`).

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```
