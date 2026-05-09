# ScanNet

Tiny ScanNet fixture obtained by subsampling real scenes from the v2 release.
For each `(version, split)` combination we keep up to 5 scenes drawn from
`metadata/scannetv2_{split}.txt`. Each scene's `_vh_clean_2.ply` is subsampled
to ≈1024 vertices via face-preserving sampling, the segment JSON is filtered to
the kept vertices, and the aggregation / metadata files are copied verbatim. The
v1 fixture reuses the v2 PLY data shipped under a v1-labelled labels file.

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/ScanNet/raw/` by
default (override with `--src-dir`). The user-local source only ships v2 scenes;
the script reuses them for v1 too.

### Raw

```bash
uv run --no-sync python scripts/generate.py raw ./raw --version v2 --split train
```

### Processed

Builds both `ScanNet/processed/{split}/` and `ScanNet/processed_20/{split}/`
caches in one go (the latter is consumed by `ScanNet20`).

```bash
uv run --no-sync python scripts/generate.py process ./raw --version v2 --split train
```

### Full regeneration

```bash
bash scripts/generate.sh
```
