# Cropping

Cropping runs in two steps: a mask transform writes one boolean per point, and `ApplyMask` drops the points outside it across every listed key.

## Which one to use

| You need                                                    | Use                             |
| ------------------------------------------------------------ | ------------------------------- |
| A region given by explicit min/max corners                  | `BoxMask`                       |
| A region a fixed distance away along every axis             | `CubeMask`                      |
| A region within a radius, measured as a real distance       | `SphereMask`                    |
| To act on a mask you already have                           | `ApplyMask`                     |
| A crop and a point cap in one step, for a training loop     | `SphereCrop`                    |
| The ego-vehicle returns gone from a LiDAR frame             | `RemoveNearOrigin`              |
| Out-of-range values pulled back in rather than dropped      | `Clamp`                         |

Parameters for each are in the [API reference](../api/transforms/transforms.md).

## Crop to a region

The three mask transforms differ only in the shape of the region. Each writes a boolean tensor to `dst_keys` and changes nothing else, so one mask can filter several keys.

=== "Object"

    ![BoxMask on an object](../assets/transforms/box_mask.png)

=== "Scene"

    ![BoxMask on a room](../assets/transforms/box_mask_scene.png)

`BoxMask` takes an axis-aligned box as a flat $(\ast\min, \ast\max)$ tuple. Use it when the region comes from a dataset's own bounds.

=== "Object"

    ![CubeMask on an object](../assets/transforms/cube_mask.png)

=== "Scene"

    ![CubeMask on a room](../assets/transforms/cube_mask_scene.png)

`CubeMask` is an $L_\infty$ ball: within `radius` of `center` along every axis independently.

=== "Object"

    ![SphereMask on an object](../assets/transforms/sphere_mask.png)

=== "Scene"

    ![SphereMask on a room](../assets/transforms/sphere_mask_scene.png)

`SphereMask` is a Euclidean ball, $\lVert x - c \rVert_2 \le r$, so `radius` means an actual distance.

`ApplyMask` then does the filtering:

```python
import torch_pointcloud.transforms as T

T.Compose([
    T.BoxMask(keys="pos", bbox=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), dst_keys="mask"),
    T.ApplyMask(keys=("pos", "color", "segment"), mask_key="mask"),
])
```

=== "Object"

    ![ApplyMask on an object](../assets/transforms/apply_mask.png)

=== "Scene"

    ![ApplyMask on a room](../assets/transforms/apply_mask_scene.png)

List every per-point key: the ones left out no longer line up with `pos`. All of them must share the leading dimension the mask was computed on. Pass `dst_keys` to write the filtered tensors under new names.

The mask is a plain boolean tensor, so `ApplyMask` also takes one you computed yourself.

## Bound memory on a large scene

`SphereCrop` does both steps at once, for a training loop. It keeps the points inside a ball and filters every listed key, and `max_nodes` caps the result at the points nearest the center.

=== "Object"

    ![SphereCrop on an object](../assets/transforms/sphere_crop.png)

=== "Scene"

    ![SphereCrop on a room](../assets/transforms/sphere_crop_scene.png)

```{.python continuation}
T.SphereCrop(pos_key="pos", radius=2.0, max_nodes=100_000, keys=("color", "segment"))
```

`center` accepts `"centroid"` (the default), `"random_point"` for a fresh crop every epoch, or an explicit 3-vector. `max_nodes` keeps a 700 000-point room inside a fixed memory budget, whatever the scan's density.

At test time, use an [inferer](../inferers/overview.md) instead of cropping: it covers the whole scene and returns one prediction per original point.

## Strip the ego-vehicle returns

A LiDAR frame contains the sensor's own mount and roof, a few thousand points at the origin that no class covers. `RemoveNearOrigin` drops them.

=== "Object"

    ![RemoveNearOrigin on an object](../assets/transforms/remove_near_origin.png)

=== "Scene"

    ![RemoveNearOrigin on a room](../assets/transforms/remove_near_origin_scene.png)

```{.python continuation}
T.RemoveNearOrigin(pos_key="pos", keys=("intensity",), radius=1.5)
```

## Keep the points, bound the values

`Clamp` moves out-of-range values onto the boundary instead of dropping their points, a thin wrapper over `torch.clamp`. Use it when the point count must stay fixed, or on a feature channel with a long tail.

=== "Object"

    ![Clamp on an object](../assets/transforms/clamp.png)

=== "Scene"

    ![Clamp on a room](../assets/transforms/clamp_scene.png)

```{.python continuation}
T.Clamp(keys="pos", min=-1.0, max=1.0)
```
