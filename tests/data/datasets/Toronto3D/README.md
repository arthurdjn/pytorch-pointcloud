# Toronto-3D

Tiny Toronto-3D fixture obtained by subsampling the four real PLY tiles
(`L001`, `L002`, `L003`, `L004`). Each file ships with 1024 randomly sampled
vertices, preserving the original CloudCompare-export schema
(`x, y, z, red, green, blue, scalar_Intensity, scalar_GPSTime, scalar_ScanAngleRank, scalar_Label`).

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/Toronto3D/raw/` by
default (override with `--src-dir`).

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```
