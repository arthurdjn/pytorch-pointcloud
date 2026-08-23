# Transforms

A transform takes **one sample** and returns one sample: a `Dict[str, Tensor]` in, a new dict out. You chain them with `Compose` and hand the result to a dataset, and everything between a file on disk and a model's forward happens here. The API is modeled on :monai: MONAI's dict transforms and :pyg: PyTorch Geometric's `Data` conventions.

```python
import torch
import torch_pointcloud.transforms as T

pos = torch.randn(2048, 3)
color = torch.rand(2048, 3)

pipeline = T.Compose([
    T.Rescale(keys="pos", method="centroid"),
    T.Shift(keys="pos", method="bbox", axes=[0, 1]),
    T.RandomSample(keys=("pos", "color"), num_samples=1024),
])

scene = pipeline({"pos": pos, "color": color})
```

## Four rules that explain the rest

1. **One sample at a time.** A transform never sees a batch, and the `batch` key is not consumed here. Everything runs before the DataLoader's collate step.
2. **Nothing is mutated.** Each transform returns a new shallow-copy dict, so the sample you passed in is still intact afterwards.
3. **No hidden combinations.** Every transform does one thing and you compose them, which is why scene centering is two explicit `Shift` calls rather than one `CenterShift`.
4. **There is a tensor-level sibling for everything.** If you already hold a tensor and want no dict at all, `torch_pointcloud.transforms.functional` has the same operation as a plain function.

## Where to start

| You are trying to                                 | Go to                             |
| -------------------------------------------------- | --------------------------------- |
| Get your cloud down to a point budget or a grid   | [Sampling](sampling.md)           |
| Center, scale or add a height feature             | [Geometric](geometric.md)         |
| Cut a scene down to a region you can afford       | [Cropping](cropping.md)           |
| Stop the model keying on lighting                 | [Color](color.md)                 |
| Decide what the model should be invariant to      | [Augmentation](augmentation.md)   |
| Build `x`, remap labels, encode detection targets | [Utilities](utilities.md)         |

If you are running a pretrained checkpoint, you may not need any of this: `info["transform"]` already carries the exact preprocessing it was trained with.

## Every transform at a glance

### Sampling and downsampling

| | Transform | Purpose |
| --- | --- | --- |
| <img src="../assets/transforms/thumbs/random_sample.png" width="220"> | [`RandomSample`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomSample) | Uniform random subsample with shared indices across keys |
| <img src="../assets/transforms/thumbs/farthest_point_sample.png" width="220"> | [`FarthestPointSample`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.FarthestPointSample) | FPS subsample (well-distributed) |
| <img src="../assets/transforms/thumbs/random_sample_face_vertices.png" width="220"> | [`RandomSampleFaceVertices`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomSampleFaceVertices) | Sample points on a mesh's faces |
| <img src="../assets/transforms/thumbs/voxelize.png" width="220"> | [`Voxelize`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Voxelize) | Voxel-grid downsample with per-voxel reduction |

### Geometry / shifting

| | Transform | Purpose |
| --- | --- | --- |
| <img src="../assets/transforms/thumbs/shift.png" width="220"> | [`Shift`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Shift) | Subtract a computed offset (`bbox`, `centroid`, or `min`) |
| <img src="../assets/transforms/thumbs/align_axis.png" width="220"> | [`AlignAxis`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.AlignAxis) | Shift one axis so its min is zero |
| <img src="../assets/transforms/thumbs/axis_min_offset.png" width="220"> | [`AxisMinOffset`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.AxisMinOffset) | Per-point offset from axis minimum (height feature) |

### Scaling / normalization

| | Transform | Purpose |
| --- | --- | --- |
| <img src="../assets/transforms/thumbs/rescale.png" width="220"> | [`Rescale`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Rescale) | Center and rescale to unit extent (`centroid` / `bbox` / `linear`) |
| <img src="../assets/transforms/thumbs/normalize.png" width="220"> | [`Normalize`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Normalize) | Per-channel $(x - \mu) / \sigma$ standardization |
| <img src="../assets/transforms/thumbs/scale.png" width="220"> | [`Scale`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Scale) | Multiply by a scalar |
| <img src="../assets/transforms/thumbs/divide.png" width="220"> | [`Divide`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Divide) | Divide by a scalar |

### Masking and filtering

| | Transform | Geometry |
| --- | --- | --- |
| <img src="../assets/transforms/thumbs/box_mask.png" width="220"> | [`BoxMask`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.BoxMask) | Axis-aligned bounding box (AABB) |
| <img src="../assets/transforms/thumbs/cube_mask.png" width="220"> | [`CubeMask`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.CubeMask) | L∞ / Chebyshev ball (hypercube) |
| <img src="../assets/transforms/thumbs/sphere_mask.png" width="220"> | [`SphereMask`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.SphereMask) | L2 / Euclidean ball |
| <img src="../assets/transforms/thumbs/apply_mask.png" width="220"> | [`ApplyMask`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ApplyMask) | Apply any precomputed mask to one or more keys |
| <img src="../assets/transforms/thumbs/remove_near_origin.png" width="220"> | [`RemoveNearOrigin`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RemoveNearOrigin) | One-shot L2 filter around origin |

### Key / dict manipulation

| | Transform | Purpose |
| --- | --- | --- |
| <img src="../assets/transforms/thumbs/cat.png" width="220"> | [`Cat`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Cat) | Concatenate multiple keys' tensors along a dim |
| <img src="../assets/transforms/thumbs/copy_items.png" width="220"> | [`CopyItems`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.CopyItems) | Clone a key's value under a new name |
| <img src="../assets/transforms/thumbs/rename_items.png" width="220"> | [`RenameItems`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RenameItems) | Move a key to a new name |
| <img src="../assets/transforms/thumbs/keep_items.png" width="220"> | [`KeepItems`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.KeepItems) | Drop everything not in a whitelist |
| <img src="../assets/transforms/thumbs/set_value.png" width="220"> | [`SetValue`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.SetValue) | Set keys to literal values |
| <img src="../assets/transforms/thumbs/subtract_key.png" width="220"> | [`SubtractKey`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.SubtractKey) | `data[k] - data[sub_k]` element-wise |
| <img src="../assets/transforms/thumbs/divide_key.png" width="220"> | [`DivideKey`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.DivideKey) | `data[k] / data[div_k]` element-wise |
| <img src="../assets/transforms/thumbs/reduce.png" width="220"> | [`Reduce`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Reduce) | Reduce a tensor along a dim (`min`/`max`/`mean`/`sum`) |
| <img src="../assets/transforms/thumbs/one_hot.png" width="220"> | [`OneHot`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.OneHot) | One-hot encode integer labels |
| <img src="../assets/transforms/thumbs/relabel.png" width="220"> | [`Relabel`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Relabel) | Remap integer labels via a lookup table |
| <img src="../assets/transforms/thumbs/abs.png" width="220"> | [`Abs`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Abs) | Element-wise absolute value |

### Type / device

| | Transform | Purpose |
| --- | --- | --- |
| <img src="../assets/transforms/thumbs/to_float.png" width="220"> | [`ToFloat`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ToFloat) | Cast tensors to float32 |
| <img src="../assets/transforms/thumbs/to_tensor.png" width="220"> | [`ToTensor`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ToTensor) | Convert lists / arrays to tensors |
| <img src="../assets/transforms/thumbs/to_device.png" width="220"> | [`ToDevice`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ToDevice) | Move to a device |
| <img src="../assets/transforms/thumbs/ones_like.png" width="220"> | [`OnesLike`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.OnesLike) | Add a key whose tensor is `torch.ones_like(...)` |

### Octree (optional, requires `ocnn`)

| | Transform | Purpose |
| --- | --- | --- |
| <img src="../assets/transforms/thumbs/build_octree.png" width="220"> | [`BuildOctree`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.BuildOctree) | Build an octree from positions |
| <img src="../assets/transforms/thumbs/octree_features.png" width="220"> | [`OctreeFeatures`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.OctreeFeatures) | Extract per-node features from an octree |

## Recipes

### Center a room the way the indoor checkpoints do

```{.python continuation}
T.Compose([
    T.Shift(keys="pos", method="bbox", axes=[0, 1]),  # XY: bbox midrange
    T.Shift(keys="pos", method="min",  axes=[2]),      # Z:  min
])
```

The two `Shift` calls touch disjoint axes, so they commute: the room is centered in XY while the floor stays at $z = 0$.

### Normalize an object to the unit sphere

```{.python continuation}
T.Compose([
    T.Rescale(keys="pos", method="centroid"),
    T.RandomSample(keys=("pos", "normal"), num_samples=1024),
])
```

### Prepare a scene for a sparse-convolution model

```{.python continuation}
T.Compose([
    T.Shift(keys="pos", method="min"),
    T.Voxelize(
        pos_key="pos", pos_reduce="grid", size=0.04,
        keys=["color", "segment"], reduce=["mean", "first"],
        dst_inverse_key="cluster",
    ),
])
```

`dst_inverse_key` stores the inverse mapping, which is what lets you project predictions back to the full-resolution cloud at evaluation time. Ask for it up front: it is easier than re-running the pipeline once you find you need it.

## Skip the dict entirely

If you already hold raw tensors and want no pipeline:

```{.python continuation}
import torch_pointcloud.transforms.functional as F

pos = F.shift(pos, method="bbox", axes=[0, 1])
pos = F.shift(pos, method="min",  axes=[2])
heights = F.axis_min_offset(pos, axis=2)
mask = F.sphere_mask(pos, center=[0.0, 0.0, 0.0], radius=2.0)
```

Every dict transform with non-trivial logic has a tensor-level sibling. See [`api/transforms/functional`](../api/transforms/functional.md).
