# KITTI

Tiny KITTI 3D object-detection fixture obtained by subsampling real frames from the object split
(3 frames, 1024 points each). The calibration and labels are copied verbatim from the real release;
only the point cloud is subsampled. The first frames that contain at least one detection-class
object are kept, so every shipped frame yields a non-empty box set.

The on-disk layout matches the real KITTI object split under `raw/` (the dataset reads from
`<root>/KITTI/raw/<split>/` and writes its `.npy` cache to `<root>/KITTI/processed[_fov]/`):

```text
raw/training/velodyne/{frame}.bin   # float32 (N, 4) = (x, y, z, intensity)
raw/training/calib/{frame}.txt      # P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo (verbatim)
raw/training/label_2/{frame}.txt    # type trunc occ alpha bbox(4) dims(h,w,l) loc(x,y,z) ry (verbatim)
raw/training/image_2/{frame}.png    # blank PNG carrying the real frame's (height, width) header
```

The real `raw/image_2/` PNGs are ~800 KB each, so each frame ships a minimal all-black PNG with the
real image dimensions instead (enough for the image-header reads, e.g. `return_calib`); the camera
field-of-view filter is covered by the unit tests.

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/KITTI` by default (override with
`--src-dir`).

```bash
uv run --no-sync python scripts/generate.py
```
