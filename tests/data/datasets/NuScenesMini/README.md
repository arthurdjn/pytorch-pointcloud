# NuScenesMini

Tiny nuScenes-mini 3D object-detection fixture obtained by subsampling real keyframes from the
`v1.0-mini` release (2 LIDAR keyframes, 3 LIDAR clouds each, 1024 points per scan). Only the point
clouds are subsampled; every metadata record is copied verbatim.

The metadata tables the loader reads are kept as a consistent slice: the chosen keyframes, the prior
sweeps reachable along their `prev` chains, and the ego poses, sensor calibration, annotations,
instances and categories those records reference. Keyframes are chosen with the fewest annotations
(so the fixture stays tiny) among those with a full sweep chain and at least one detection-class
object.

The on-disk layout matches the extracted `v1.0-mini`:

```text
raw/v1.0-mini/*.json              # ego_pose, calibrated_sensor, category, instance,
                                  # sample_annotation, sample_data (consistent subset, verbatim)
raw/samples/LIDAR_TOP/*.pcd.bin   # float32 (N, 5) = (x, y, z, intensity, ring) keyframes
raw/sweeps/LIDAR_TOP/*.pcd.bin    # float32 (N, 5) prior sweeps
```

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/NuScenesMini` by default (override with
`--src-dir`).

```bash
uv run --no-sync python scripts/generate.py
```
