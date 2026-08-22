# Sampling

Sampling transforms control how many points a cloud has and how they are distributed. Each transform is shown on a single object and a real ScanNet room; the **Object** / **Scene** tabs switch together across the page.

| Transform                                                                                                                     | Functional                    | Purpose                                                  |
| ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------- |
| [`RandomSample`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomSample)                         | `random_sample`               | Uniform random subsample with shared indices across keys |
| [`FarthestPointSample`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.FarthestPointSample)           | `farthest_point_sample`       | FPS subsample (well-distributed)                         |
| [`Voxelize`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Voxelize)                                 | (in `utils.ops`)              | Voxel-grid downsample with per-voxel reduction           |
| [`Quantize`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Quantize)                                 | `quantize`                    | Integer grid coordinates, cloud kept at full resolution  |
| [`HardVoxelize`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.HardVoxelize)                         | (in `utils.voxelization`)     | Fixed-capacity voxel tensors for grid detectors          |
| [`RandomSampleFaceVertices`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomSampleFaceVertices) | `random_sample_face_vertices` | Sample points on a mesh's faces                          |
| [`ShufflePoint`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ShufflePoint)                         | `shuffle_indices`             | Randomly permute point order                             |

For random point dropout as a training augmentation, see [`RandomDropout`](augmentation.md#randomdropout).

## RandomSample

Uniform random subsample to `num_samples`; all listed keys share the same indices, so per-point correspondence is preserved.

=== "Object"

    ![RandomSample on an object](../assets/transforms/random_sample.png)

=== "Scene"

    ![RandomSample on a room](../assets/transforms/random_sample_scene.png)

```python
import torch_pointcloud.transforms as T

T.RandomSample(keys=("pos", "color"), num_samples=1024)
```

## FarthestPointSample

Repeatedly picks the point farthest from those already chosen: slower than random sampling, but evenly distributed. The same FPS convention used inside PointNet++ and PointNeXt.

=== "Object"

    ![FarthestPointSample on an object](../assets/transforms/farthest_point_sample.png)

=== "Scene"

    ![FarthestPointSample on a room](../assets/transforms/farthest_point_sample_scene.png)

```{.python continuation}
T.FarthestPointSample(pos_key="pos", keys=("color",), num_samples=1024)
```

## Voxelize

Bins points into a grid and keeps one representative per occupied voxel, with a per-key reduction (`mean` color, `first` label, ...). `dst_inverse_key` stores the map back to full resolution.

=== "Object"

    ![Voxelize on an object](../assets/transforms/voxelize.png)

=== "Scene"

    ![Voxelize on a room (voxels keep mean color)](../assets/transforms/voxelize_scene.png)

```{.python continuation}
T.Compose([
    T.Shift(keys="pos", method="min"),
    T.Voxelize(
        pos_key="pos", pos_reduce="grid", size=0.04,
        keys=["color", "segment"], reduce=["mean", "first"],
        dst_inverse_key="inverse",
    ),
])
```

`dst_inverse_key` lets you project voxel-level predictions back to the full-resolution cloud with `preds_full = preds_voxel[inverse]`: the standard prep for sparse-conv segmentation.

## Quantize

Writes the integer grid coordinate $\lfloor p / s \rfloor$ of every point, shifted so the per-axis minimum is $0$, and keeps the cloud at full resolution. Nothing is reduced: points sharing a voxel keep their own rows and get identical coordinates, which is what the figure's note counts (1024 rows landing on 35 distinct nodes).

=== "Object"

    ![Quantize on an object](../assets/transforms/quantize.png)

=== "Scene"

    ![Quantize on a room](../assets/transforms/quantize_scene.png)

```{.python continuation}
T.Quantize(keys="pos", size=0.02, dst_keys="pos_grid")
```

This is how a voxel-partition evaluation feeds a sparse model every raw point, and how a test-time view recomputes grid coordinates after rotating or scaling the positions. Use `Voxelize` instead when you want one point per voxel.

## HardVoxelize

Builds the fixed-capacity voxel stack that grid detectors (PointPillars, SECOND) consume, moving that step out of the model and into the pipeline. Points are binned over `point_cloud_range`, at most `max_num_points` are kept per voxel and at most `max_num_voxels` per scene; the figure mutes the points that overflow their voxel.

=== "Object"

    ![HardVoxelize on an object](../assets/transforms/hard_voxelize.png)

=== "Scene"

    ![HardVoxelize on a room](../assets/transforms/hard_voxelize_scene.png)

```{.python continuation}
T.HardVoxelize(
    pos_key="pos",
    feat_key="x",
    voxel_size=(0.16, 0.16, 4.0),
    point_cloud_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
    max_num_points=32,
    max_num_voxels=40000,
)
```

It adds three keys and keeps `pos` / `x`: the per-voxel point stack $(V, \text{max\_num\_points}, 3 + C)$ at `voxel_key`, the integer grid indices $(V, 3)$ in $(z, y, x)$ order at `pos_voxel_key`, and the per-voxel point counts $(V,)$ at `num_points_key`.

## RandomSampleFaceVertices

Samples points on mesh faces: how a ModelNet mesh becomes a point cloud. Surface normals are stored under `normal_key`.

=== "Object"

    ![Points sampled on a mesh surface](../assets/transforms/random_sample_face_vertices.png)

```{.python continuation}
T.RandomSampleFaceVertices(keys="pos", face_key="face", num_samples=2048)
```

## ShufflePoint

Permutes the point order, with the same permutation on every listed key so per-point correspondence survives. The figure colors points by their row index before shuffling, so the gradient scrambles while the shape stays identical.

=== "Object"

    ![ShufflePoint on an object](../assets/transforms/shuffle_point.png)

=== "Scene"

    ![ShufflePoint on a room](../assets/transforms/shuffle_point_scene.png)

```{.python continuation}
T.ShufflePoint(keys=("pos", "color"), p=1.0)
```

Use it to break a structural ordering in the input before a transform whose result depends on that order, such as `Slice` or the `first` reduction of `Voxelize`.
