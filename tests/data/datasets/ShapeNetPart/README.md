# ShapeNetPart

Tiny ShapeNetPart fixture obtained by subsampling real per-object `.txt` files
from the original release. For each of the 16 categories we keep up to 4 objects
and 1024 points each. The split files (`train`/`val`/`test`) are filtered to the
ids actually shipped.

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/ShapeNetPart/raw/`
by default (override with `--src-dir`).

```bash
uv run --no-sync python scripts/generate.py raw ./raw
uv run --no-sync python scripts/generate.py process ./raw
```

> [!NOTE]
> The processed files (`pos.npy`, `normal.npy`, `segment.npy`, `category.npy`,
> `offset.npy`) land under `processed/{split}/`.
