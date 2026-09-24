# Transforms

A transform takes one sample dict and returns a new one. `Compose` chains them, and a dataset applies the chain to every sample it loads. The API follows :monai: MONAI's dict transforms and :pyg: PyTorch Geometric's `Data` conventions.

!!! question "Why a dict-based API?"
    The dict-based API keeps composition flexible and makes the inputs and outputs of each transform explicit.

    :pyg: PyTorch Geometric transforms, in contrast, take a `Data` object. That is limiting once you want to carry extra keys such as intensity or color, because every transform then has to handle every attribute the object holds, defaulting the ones you did not provide to `None`.

    Transforms are designed to manipulate specific keys of the input data, which makes the operations explicit and easier to compose.

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

!!! tip "Atomic operations"
    Each transform is designed as an atomic operation to make it easier to compose and reuse.

!!! note "Pretrained checkpoints"
    A pretrained checkpoint carries its own preprocessing in `info["transform"]`, for inference on its dataset.

## Sampling and downsampling

|                                                                                     | Transform                                                                                                                 | Description                                              |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| <img src="../assets/transforms/thumbs/random_sample.png" width="220">               | [`RandomSample`](../api/transforms/sampling.md#torch_pointcloud.transforms.sampling.RandomSample)                         | Uniform random subsample with shared indices across keys |
| <img src="../assets/transforms/thumbs/farthest_point_sample.png" width="220">       | [`FarthestPointSample`](../api/transforms/sampling.md#torch_pointcloud.transforms.sampling.FarthestPointSample)           | FPS subsample (well-distributed)                         |
| <img src="../assets/transforms/thumbs/random_sample_face_vertices.png" width="220"> | [`RandomSampleFaceVertices`](../api/transforms/sampling.md#torch_pointcloud.transforms.sampling.RandomSampleFaceVertices) | Sample points on a mesh's faces                          |
| <img src="../assets/transforms/thumbs/voxelize.png" width="220">                    | [`Voxelize`](../api/transforms/voxelization.md#torch_pointcloud.transforms.voxelization.Voxelize)                         | Voxel-grid downsample with per-voxel reduction           |

### Sampling keys

Any transform that changes the number of points keeps `pos`, `x`, `segment` and `batch` aligned at the new resolution and records how to get back:

| Key              | Shape                  | Written by                                                                                                | Meaning                                                                 |
| ---------------- | ---------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `origin_pos`     | $(N_\text{origin}, 3)$ | a `CopyItems` step in every registered pipeline                                                           | the source cloud, in the same frame as `pos`                            |
| `origin_segment` | $(N_\text{origin},)$   | the same step, when the pipeline carries labels                                                           | the source labels, in the model's label space                           |
| `inverse`        | $(N_\text{origin},)$   | `Voxelize`, `DivisiblePad` through `dst_inverse_key`                                                      | source row to predictor row: `preds[inverse]` scores at full resolution |
| `index`          | $(N,)$                 | the selection samplers (`FarthestPointSample`, `RandomSample`, `SphereCrop`, ...) through `dst_index_key` | predictor row to source row: `origin_pos[index]` is `pos`               |

Chained steps compose these maps, so they always address the outermost source. Registered pipelines set the keys; set them on your own samplers to get the maps.

## Geometry / shifting

|                                                                         | Transform                                                                                           | Description                                               |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| <img src="../assets/transforms/thumbs/shift.png" width="220">           | [`Shift`](../api/transforms/geometry.md#torch_pointcloud.transforms.geometry.Shift)                 | Subtract a computed offset (`bbox`, `centroid`, or `min`) |
| <img src="../assets/transforms/thumbs/axis_min_offset.png" width="220"> | [`AxisMinOffset`](../api/transforms/geometry.md#torch_pointcloud.transforms.geometry.AxisMinOffset) | Per-point offset from axis minimum (height feature)       |

## Scaling / normalization

|                                                                   | Transform                                                                                 | Description                                                                                |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| <img src="../assets/transforms/thumbs/rescale.png" width="220">   | [`Rescale`](../api/transforms/scaling.md#torch_pointcloud.transforms.scaling.Rescale)     | Center and rescale to unit extent (`centroid` / `bbox` / `centroid_extent` / `min_sphere`) |
| <img src="../assets/transforms/thumbs/normalize.png" width="220"> | [`Normalize`](../api/transforms/scaling.md#torch_pointcloud.transforms.scaling.Normalize) | Per-channel $(x - \mu) / \sigma$ standardization                                           |

## Masking and filtering

|                                                                            | Transform                                                                                               | Geometry                                       |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| <img src="../assets/transforms/thumbs/box_mask.png" width="220">           | [`BoxMask`](../api/transforms/masking.md#torch_pointcloud.transforms.masking.BoxMask)                   | Axis-aligned bounding box (AABB)               |
| <img src="../assets/transforms/thumbs/cube_mask.png" width="220">          | [`CubeMask`](../api/transforms/masking.md#torch_pointcloud.transforms.masking.CubeMask)                 | L∞ / Chebyshev ball (hypercube)                |
| <img src="../assets/transforms/thumbs/sphere_mask.png" width="220">        | [`SphereMask`](../api/transforms/masking.md#torch_pointcloud.transforms.masking.SphereMask)             | L2 / Euclidean ball                            |
| <img src="../assets/transforms/thumbs/apply_mask.png" width="220">         | [`ApplyMask`](../api/transforms/masking.md#torch_pointcloud.transforms.masking.ApplyMask)               | Apply any precomputed mask to one or more keys |
| <img src="../assets/transforms/thumbs/remove_near_origin.png" width="220"> | [`RemoveNearOrigin`](../api/transforms/masking.md#torch_pointcloud.transforms.masking.RemoveNearOrigin) | One-shot L2 filter around origin               |

## Utilities

|                                                                        | Transform                                                                                     | Description                                            |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| <img src="../assets/transforms/thumbs/cat.png" width="220">            | [`Cat`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.Cat)                     | Concatenate multiple keys' tensors along a dim         |
| <img src="../assets/transforms/thumbs/copy_items.png" width="220">     | [`CopyItems`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.CopyItems)         | Clone a key's value under a new name                   |
| <img src="../assets/transforms/thumbs/rename_items.png" width="220">   | [`RenameItems`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.RenameItems)     | Move a key to a new name                               |
| <img src="../assets/transforms/thumbs/keep_items.png" width="220">     | [`KeepItems`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.KeepItems)         | Drop everything not in a whitelist                     |
| <img src="../assets/transforms/thumbs/set_value.png" width="220">      | [`SetValue`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.SetValue)           | Set keys to literal values                             |
| <img src="../assets/transforms/thumbs/reduce.png" width="220">         | [`Reduce`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.Reduce)               | Reduce a tensor along a dim (`min`/`max`/`mean`/`sum`) |
| <img src="../assets/transforms/thumbs/one_hot.png" width="220">        | [`OneHot`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.OneHot)               | One-hot encode integer labels                          |
| <img src="../assets/transforms/thumbs/relabel.png" width="220">        | [`Relabel`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.Relabel)             | Remap integer labels via a lookup table                |
| <img src="../assets/transforms/thumbs/scale.png" width="220">          | [`Scale`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.Scale)                 | Multiply by a scalar                                   |
| <img src="../assets/transforms/thumbs/divide.png" width="220">         | [`Divide`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.Divide)               | Divide by a scalar                                     |
| <img src="../assets/transforms/thumbs/divide_items.png" width="220">   | [`DivideItems`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.DivideItems)     | `data[k] / data[div_k]` element-wise                   |
| <img src="../assets/transforms/thumbs/subtract_items.png" width="220"> | [`SubtractItems`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.SubtractItems) | `data[k] - data[sub_k]` element-wise                   |
| <img src="../assets/transforms/thumbs/abs.png" width="220">            | [`Abs`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.Abs)                     | Element-wise absolute value                            |
| <img src="../assets/transforms/thumbs/to_float.png" width="220">       | [`ToFloat`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.ToFloat)             | Cast tensors to float32                                |
| <img src="../assets/transforms/thumbs/to_tensor.png" width="220">      | [`ToTensor`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.ToTensor)           | Convert lists / arrays to tensors                      |
| <img src="../assets/transforms/thumbs/to_device.png" width="220">      | [`ToDevice`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.ToDevice)           | Move to a device                                       |
| <img src="../assets/transforms/thumbs/ones_like.png" width="220">      | [`OnesLike`](../api/transforms/utils.md#torch_pointcloud.transforms.utils.OnesLike)           | Add a key whose tensor is `torch.ones_like(...)`       |

## Octree (optional, requires `ocnn`)

|                                                                         | Transform                                                                                         | Description                              |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| <img src="../assets/transforms/thumbs/build_octree.png" width="220">    | [`BuildOctree`](../api/transforms/octree.md#torch_pointcloud.transforms.octree.BuildOctree)       | Build an octree from positions           |
| <img src="../assets/transforms/thumbs/octree_features.png" width="220"> | [`OctreeFeatures`](../api/transforms/octree.md#torch_pointcloud.transforms.octree.OctreeFeatures) | Extract per-node features from an octree |
