# Transforms

`torch-pointcloud.transforms` is the preprocessing layer. It is modelled on :monai: MONAI's dict transforms and :pyg: PyTorch Geometric's `Data` conventions: every transform operates on a `Dict[str, Tensor]` representing a **single scene** (one sample, pre-collate), and transforms chain via `Compose`.

```python
import torch_pointcloud.transforms as T

pipeline = T.Compose([
    T.Rescale(keys="pos", method="centroid"),
    T.Shift(keys="pos", method="bbox", axes=[0, 1]),
    T.RandomSample(keys=("pos", "color"), num_samples=1024),
])

scene = pipeline({"pos": pos, "color": color})
```

## Design

1. **Single-scene contract.** Transforms operate on one sample. The `batch` key is not consumed here. Use `Compose` before the DataLoader's collate step.
2. **Non-mutating.** Each transform returns a new shallow-copy dict. The input dict is never modified.
3. **Composable.** Chain primitives with `Compose`. There are no hidden combo transforms (the old `CenterShift` was removed in favour of two explicit `Shift` calls).
4. **Functional layer mirrors the class layer.** Every non-trivial transform has a tensor-level functional sibling under `torch_pointcloud.transforms.functional`. Class transforms are the dict wrappers; functions are for users who already have a tensor.

## Cheat sheet

### Sampling and downsampling

| Transform | Functional | Purpose |
| --- | --- | --- |
| [`RandomSample`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomSample) | `random_sample` | Uniform random subsample with shared indices across keys |
| [`FarthestPointSample`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.FarthestPointSample) | `farthest_point_sample` | FPS subsample (well-distributed) |
| [`RandomSampleFaceVertices`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomSampleFaceVertices) | `random_sample_face_vertices` | Sample points on a mesh's faces |
| [`Voxelize`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Voxelize) | (in `utils.ops`) | Voxel-grid downsample with per-voxel reduction |

### Geometry / shifting

| Transform | Functional | Purpose |
| --- | --- | --- |
| [`Shift`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Shift) | `shift` | Subtract a computed offset (`bbox`, `centroid`, or `min`) |
| [`AlignAxis`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.AlignAxis) | (use `shift(method="min", axes=[k])`) | Shift one axis so its min is zero |
| [`AxisMinOffset`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.AxisMinOffset) | `axis_min_offset` | Per-point offset from axis minimum (height feature) |

### Scaling / normalization

| Transform | Functional | Purpose |
| --- | --- | --- |
| [`Rescale`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Rescale) | `rescale` | Center and rescale to unit extent (`centroid` / `bbox` / `linear`) |
| [`Normalize`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Normalize) | `normalize` | Per-channel $(x - \mu) / \sigma$ standardization |
| [`Scale`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Scale) | (`x * s`) | Multiply by a scalar |
| [`Divide`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Divide) | (`x / d`) | Divide by a scalar |

### Masking and filtering

| Transform | Functional | Geometry |
| --- | --- | --- |
| [`BoxMask`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.BoxMask) | `box_mask` | Axis-aligned bounding box (AABB) |
| [`CubeMask`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.CubeMask) | `cube_mask` | L∞ / Chebyshev ball (hypercube) |
| [`SphereMask`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.SphereMask) | `sphere_mask` | L2 / Euclidean ball |
| [`ApplyMask`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ApplyMask) | `apply_mask` | Apply any precomputed mask to one or more keys |
| [`RemoveNearOrigin`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RemoveNearOrigin) | `remove_near_origin` | One-shot L2 filter around origin |

### Key / dict manipulation

| Transform | Purpose |
| --- | --- |
| [`Cat`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Cat) | Concatenate multiple keys' tensors along a dim |
| [`CopyItems`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.CopyItems) | Clone a key's value under a new name |
| [`RenameItems`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RenameItems) | Move a key to a new name |
| [`KeepItems`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.KeepItems) | Drop everything not in a whitelist |
| [`SetValue`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.SetValue) | Set keys to literal values |
| [`SubtractKey`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.SubtractKey) | `data[k] - data[sub_k]` element-wise |
| [`DivideKey`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.DivideKey) | `data[k] / data[div_k]` element-wise |
| [`Reduce`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Reduce) | Reduce a tensor along a dim (`min`/`max`/`mean`/`sum`) |
| [`OneHot`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.OneHot) | One-hot encode integer labels |
| [`Relabel`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Relabel) | Remap integer labels via a lookup table |
| [`Abs`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Abs) | Element-wise absolute value |

### Type / device

| Transform | Purpose |
| --- | --- |
| [`ToFloat`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ToFloat) | Cast tensors to float32 |
| [`ToTensor`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ToTensor) | Convert lists / arrays to tensors |
| [`ToDevice`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.ToDevice) | Move to a device |
| [`OnesLike`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.OnesLike) | Add a key whose tensor is `torch.ones_like(...)` |

### Octree (optional, requires `ocnn`)

| Transform | Purpose |
| --- | --- |
| [`BuildOctree`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.BuildOctree) | Build an octree from positions |
| [`OctreeFeatures`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.OctreeFeatures) | Extract per-node features from an octree |

## Recipes

### Pointcept-style scene centering

```python
T.Compose([
    T.Shift(keys="pos", method="bbox", axes=[0, 1]),  # XY: bbox midrange
    T.Shift(keys="pos", method="min",  axes=[2]),      # Z:  min
])
```

The two `Shift` calls touch disjoint axes, so they commute and produce the same offset as the historical `CenterShift(apply_z=True)`.

### Unit-sphere normalization (ModelNet-style)

```python
T.Compose([
    T.Rescale(keys="pos", method="centroid"),
    T.RandomSample(keys=("pos", "normal"), num_samples=1024),
])
```

### Voxel-prep for sparse-conv segmentation

```python
T.Compose([
    T.Shift(keys="pos", method="min"),
    T.Voxelize(
        pos_key="pos", pos_reduce="grid", size=0.04,
        keys=["color", "segment"], reduce=["mean", "first"],
        cluster_key="cluster",
    ),
])
```

`cluster_key` stores the inverse mapping so you can project predictions back to the full-resolution cloud at evaluation time.

## Going lower-level

If you already have raw tensors and don't want a dict pipeline:

```python
import torch_pointcloud.transforms.functional as F

pos = F.shift(pos, method="bbox", axes=[0, 1])
pos = F.shift(pos, method="min",  axes=[2])
heights = F.axis_min_offset(pos, axis=2)
mask = F.sphere_mask(pos, center=[0.0, 0.0, 0.0], radius=2.0)
```

Every dict transform with non-trivial logic has a tensor-level sibling. See [`api/transforms/functional`](../api/transforms/functional.md).
