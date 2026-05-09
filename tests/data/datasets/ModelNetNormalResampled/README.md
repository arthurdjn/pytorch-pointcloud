# ModelNetNormalResampled

Tiny ModelNetNormalResampled fixture obtained by subsampling real `.txt` point
clouds from the original release. For each ModelNet variant (10 and 40) and
split (train / test) we keep 2 objects per class (≈1024 points each); the split
`.txt` files are rewritten to reference only the ids actually shipped.

## Generation

`scripts/generate.py` reads from
`$TORCH_POINTCLOUD_DATA_DIR/ModelNetNormalResampled/raw/` by default (override
with `--src-dir`).

```bash
uv run --no-sync python scripts/generate.py raw ./raw
uv run --no-sync python scripts/generate.py process ./raw
```

> [!NOTE]
> The processed `modelnet{10,40}_{train,test}.dat` land under `processed/`.
