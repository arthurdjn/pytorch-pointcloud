# S3DIS

Tiny S3DIS fixture obtained by subsampling real per-annotation `.txt` files from
the original release. For each of the six areas we keep 2 rooms and 64 points per
annotation; the room-level concatenated file (`{room}/{room}.txt`) and
`{Area}_alignmentAngle.txt` are written so the standard loader works unchanged.

The HDF5 fixture under `indoor3d_sem_seg_hdf5_data/` is the real Pointcept release
(4096 points per block) and is consumed by `S3DISHdf5` in the pretrained tests.

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/S3DIS/raw/` by
default (override with `--src-dir`).

```bash
uv run --no-sync python scripts/generate.py raw ./raw
uv run --no-sync python scripts/generate.py process ./raw
```

> [!NOTE]
> The processed files land under `processed_aligned/{Area}/`.
