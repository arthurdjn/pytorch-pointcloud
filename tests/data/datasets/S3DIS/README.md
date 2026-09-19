# S3DIS

Synthetic S3DIS: 2 box rooms per area (64 points per annotation, all 13 classes plus a `stairs` annotation), plus the
`processed_aligned/` cache. The `indoor3d_sem_seg_hdf5_data/` blocks are random too and are not written by the script.

```bash
uv run --no-sync python scripts/generate.py raw ./raw
uv run --no-sync python scripts/generate.py process ./raw
```
