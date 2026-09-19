# ScanObjectNN

Synthetic ScanObjectNN: one `.h5` of 2 objects (1024 points) per (partition, background, split, variant), plus the
processed `.npz` caches. `main_split_nobg/test_objectdataset.h5` holds one object per class.

```bash
uv run --no-sync python scripts/generate.py raw ./raw
uv run --no-sync python scripts/generate.py process ./raw
```
