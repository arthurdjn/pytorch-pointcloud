# ModelNet10

Tiny ModelNet10 fixture obtained by subsampling real `.off` meshes from the
original release. For each of the 10 classes we keep 2 train and 2 test meshes,
each subsampled to ≈1024 vertices via face-preserving sampling so that the
remaining triangles (and therefore vertex normals) are well-defined.

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/ModelNet10/raw/` by
default (override with `--src-dir`).

```bash
uv run --no-sync python scripts/generate.py raw ./raw --variant 10
uv run --no-sync python scripts/generate.py process ./raw --variant 10
```

> [!NOTE]
> The processed `train.pt` / `test.pt` land under `processed/`.
