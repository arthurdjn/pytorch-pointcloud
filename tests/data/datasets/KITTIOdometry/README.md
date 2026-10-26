# KITTI odometry

Synthetic KITTI odometry: one sequence per split (`00`, `06`, `08`) of 2 or 3 scans in the velodyne release layout
(`sequences/<seq>/velodyne/*.bin`), with the GeoTransformer pair lists under `metadata/`.

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```
