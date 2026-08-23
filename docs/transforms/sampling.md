# Sampling

Sampling transforms set how many points a model reads. A ModelNet mesh carries no points, a ScanNet room carries 700 000, and a grid detector reads a fixed-capacity voxel stack.

## Which one to use

| You need                                                  | Use                                                |
| --------------------------------------------------------- | -------------------------------------------------- |
| A fixed point count, as fast as possible                  | `RandomSample`                                     |
| A fixed point count, evenly spread over the shape         | `FarthestPointSample`                              |
| One point per voxel, at a resolution you choose           | `Voxelize`                                         |
| Every point kept, each tagged with its grid coordinate    | `Quantize`                                         |
| The fixed-capacity voxel stack a grid detector reads      | `HardVoxelize`                                     |
| Points at all, starting from a triangle mesh              | `RandomSampleFaceVertices`                         |
| The input order broken before an order-dependent step     | `ShufflePoint`                                     |
| Points dropped at random, as a training augmentation      | [`RandomDropout`](augmentation.md#vary-the-density)   |

Parameters for each are in the [API reference](../api/transforms/transforms.md).

## Give an object model its point budget

Object models train on a fixed count, usually 1024 or 2048 points. `RandomSample` draws them at random. `FarthestPointSample` picks each next point farthest from the ones already chosen, as PointNet++ and PointNeXt do. Both share their indices across every listed key, so `color` and `segment` stay aligned with `pos`.

=== "Object"

    ![RandomSample on an object](../assets/transforms/random_sample.png)

=== "Scene"

    ![RandomSample on a room](../assets/transforms/random_sample_scene.png)

```python
import torch_pointcloud.transforms as T

fast = T.RandomSample(keys=("pos", "color"), num_samples=1024)
even = T.FarthestPointSample(pos_key="pos", keys=("color",), num_samples=1024)
```

=== "Object"

    ![FarthestPointSample on an object](../assets/transforms/farthest_point_sample.png)

=== "Scene"

    ![FarthestPointSample on a room](../assets/transforms/farthest_point_sample_scene.png)

FPS costs more and covers the shape better. Most classification and part-segmentation checkpoints use it. `RandomSample` is enough in a training loop over large scenes, where every epoch draws a new subset.

## Feed a sparse-convolution scene model

Scene models work on a voxel grid, so downsample by resolution instead of by count. `Voxelize` bins the points and keeps one per occupied voxel. Each key is reduced its own way: `mean` for color, `first` for a label.

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

Pass `dst_inverse_key` to score the result at full resolution. It stores the map back to the original points, so `preds_full = preds_voxel[inverse]` gives one prediction per point.

![The same room voxelized at 3, 6, 12 and 24 cm](../assets/animations/voxelize.webp)

Voxel size sets the resolution of the pipeline. 2-4 cm keeps furniture legible, and every doubling divides the point count by roughly eight.

## Keep every point but give it a grid coordinate

`Quantize` writes the integer grid coordinate $\lfloor p / s \rfloor$ of every point, shifted so the per-axis minimum is $0$, and reduces nothing. Points sharing a voxel keep their own rows and get identical coordinates.

=== "Object"

    ![Quantize on an object](../assets/transforms/quantize.png)

=== "Scene"

    ![Quantize on a room](../assets/transforms/quantize_scene.png)

```{.python continuation}
T.Quantize(keys="pos", size=0.02, dst_keys="pos_grid")
```

Use `Quantize` when a sparse model must see every raw point, as in a voxel-partition evaluation, or when a test-time view recomputes grid coordinates after a rotation or a scaling. Use `Voxelize` when one point per voxel is enough.

## Prepare a driving frame for a grid detector

`HardVoxelize` builds the fixed-capacity voxel stack PointPillars and SECOND read, so that step stays in the pipeline instead of the model. Points are binned over `point_cloud_range`, at most `max_num_points` survive per voxel and at most `max_num_voxels` per scene.

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

It keeps `pos` and `x` and adds three keys: the per-voxel point stack $(V, \text{max\_num\_points}, 3 + C)$ at `voxel_key`, the integer grid indices $(V, 3)$ in $(z, y, x)$ order at `pos_voxel_key`, and the per-voxel point counts $(V,)$ at `num_points_key`. Those are the four tensors a voxel detector's forward takes, in that order.

## Turn a mesh into a point cloud

`ModelNet10` and `ModelNet40` ship triangle meshes. `RandomSampleFaceVertices` samples points on the faces and writes the surface normals to `normal_key`.

=== "Object"

    ![Points sampled on a mesh surface](../assets/transforms/random_sample_face_vertices.png)

```{.python continuation}
T.RandomSampleFaceVertices(keys="pos", face_key="face", num_samples=2048)
```

The sampling is random, so two runs give different clouds and a published score is hard to reproduce exactly. The preprocessed `ModelNetNormalResampled` release avoids that; see [Classification datasets](../datasets/classification.md).

## Break an ordering you did not intend to rely on

`ShufflePoint` permutes the point order, applying the same permutation to every listed key so correspondence survives.

=== "Object"

    ![ShufflePoint on an object](../assets/transforms/shuffle_point.png)

=== "Scene"

    ![ShufflePoint on a room](../assets/transforms/shuffle_point_scene.png)

```{.python continuation}
T.ShufflePoint(keys=("pos", "color"), p=1.0)
```

Use it before any step that depends on row order, such as `Slice` or the `first` reduction of `Voxelize`. A file sorted by scan line would otherwise leak that order into the labels.
