# ScanObjectNN

Tiny ScanObjectNN fixture obtained by subsampling real `.h5` archives from the
original release. Each (split, background, train, variant) combination is
preserved on disk with two real objects (one per class for the file used by the
per-class label test) downsampled to 1024 points.

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/ScanObjectNN/raw/`
by default (override with `--src-dir`).

### Raw

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```

### Processed

```bash
uv run --no-sync python scripts/generate.py process ./raw
```

> [!NOTE]
> The processed `.npz` files land under `processed/`.

### Full regeneration

```bash
bash scripts/generate.sh
```
