# Geometric

Geometric transforms position and scale a cloud deterministically: the canonical first steps of any pipeline. Each transform is shown on a single object and a real ScanNet room; the **Object** / **Scene** tabs switch together across the page.

| Transform                                                                                               | Functional                            | Purpose                                                            |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| [`Shift`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Shift)                 | `shift`                               | Subtract a computed offset (`bbox`, `centroid`, or `min`)          |
| [`AlignAxis`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.AlignAxis)         | (use `shift(method="min", axes=[k])`) | Shift one axis so its min is zero                                  |
| [`AxisMinOffset`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.AxisMinOffset) | `axis_min_offset`                     | Per-point offset from axis minimum (height feature)                |
| [`Rescale`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Rescale)             | `rescale`                             | Center and rescale to unit extent (`centroid` / `bbox` / `linear`) |
| [`Normalize`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Normalize)         | `normalize`                           | Per-channel $(x - \mu) / \sigma$ standardization                   |
| [`Scale`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Scale)                 | (`x * s`)                             | Multiply by a scalar                                               |
| [`Divide`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Divide)               | (`x / d`)                             | Divide by a scalar                                                 |
| [`SubtractKey`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.SubtractKey)     | (`x - data[k]`)                       | Subtract the tensor held under another key                         |
| [`BBoxCenter`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.BBoxCenter)       | (`(min + max) / 2`)                   | Midpoint of a stored flat bbox                                     |
| [`Abs`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.Abs)                     | `abs`                                 | Element-wise absolute value                                        |

## Shift

`Shift` takes a `method`. The arrow is the shift vector: the reference point (centroid, bbox-centre, or min-corner) is moved onto the origin.

### method="centroid"

Subtracts the mean: the centroid lands on the origin.

=== "Object"

    ![Shift method=centroid on an object](../assets/transforms/shift_centroid.png)

=== "Scene"

    ![Shift method=centroid on a room](../assets/transforms/shift_centroid_scene.png)

```python
import torch_pointcloud.transforms as T

T.Shift(keys="pos", method="centroid")
```

### method="bbox"

Subtracts the bounding-box midrange $(\min + \max) / 2$.

=== "Object"

    ![Shift method=bbox on an object](../assets/transforms/shift_bbox.png)

=== "Scene"

    ![Shift method=bbox on a room](../assets/transforms/shift_bbox_scene.png)

```{.python continuation}
T.Shift(keys="pos", method="bbox")
```

### method="min"

Subtracts the per-axis minimum: the cloud's corner lands on the origin.

=== "Object"

    ![Shift method=min on an object](../assets/transforms/shift_min.png)

=== "Scene"

    ![Shift method=min on a room](../assets/transforms/shift_min_scene.png)

```{.python continuation}
T.Shift(keys="pos", method="min")
```

### axes

`axes` limits recentering to a subset of axes; the others keep their offset (the gray dot is the origin).

=== "Object"

    ![Shift recentering axis subsets on an object](../assets/transforms/shift_axes.png)

=== "Scene"

    ![Shift recentering axis subsets on a room](../assets/transforms/shift_axes_scene.png)

Two `Shift` calls on disjoint axes commute. This is the Pointcept-style scene centering recipe, equivalent to the historical `CenterShift(apply_z=True)`:

```{.python continuation}
T.Compose([
    T.Shift(keys="pos", method="bbox", axes=[0, 1]),  # XY: bbox midrange
    T.Shift(keys="pos", method="min",  axes=[2]),     # Z:  min
])
```

## AlignAxis

Shifts a single axis so its minimum lands on zero and leaves the other axes where they are. With `dim=2` a room's lowest point sits on $z = 0$ (the gray dot is the origin).

=== "Object"

    ![AlignAxis on an object](../assets/transforms/align_axis.png)

=== "Scene"

    ![AlignAxis on a room](../assets/transforms/align_axis_scene.png)

```{.python continuation}
T.AlignAxis(keys="pos", dim=2)
```

Same offset as `T.Shift(keys="pos", method="min", axes=[2])`, kept as a one-axis preset.

## AxisMinOffset

Writes a per-point scalar instead of moving the cloud: each point's offset above the minimum along `axis`, its height over the local floor. A $(N, 3)$ input gives a $(N, 1)$ output, so it pairs with `Cat` to append a height channel to the features. In the figure the positions are untouched and colored by the value written to `height`.

=== "Object"

    ![AxisMinOffset on an object](../assets/transforms/axis_min_offset.png)

=== "Scene"

    ![AxisMinOffset on a room](../assets/transforms/axis_min_offset_scene.png)

```{.python continuation}
T.AxisMinOffset(keys="pos", dst_keys="height", axis=2)
```

`quantile` replaces the strict minimum with an empirical quantile, an outlier-robust floor estimate: `quantile=0.0099` reproduces VoteNet's height feature.

## Rescale

`Rescale` takes a `method`; the blue box and star mark the bbox and centroid it normalizes.

### method="centroid"

Centers on the centroid and divides by the max distance to it: the cloud fits the unit sphere (ModelNet-style).

=== "Object"

    ![Rescale method=centroid on an object](../assets/transforms/rescale_centroid.png)

=== "Scene"

    ![Rescale method=centroid on a room](../assets/transforms/rescale_centroid_scene.png)

```{.python continuation}
T.Rescale(keys="pos", method="centroid")
```

### method="bbox"

Centers on the bbox midpoint and divides by half the longest extent: the longest axis spans $[-1, 1]$.

=== "Object"

    ![Rescale method=bbox on an object](../assets/transforms/rescale_bbox.png)

=== "Scene"

    ![Rescale method=bbox on a room](../assets/transforms/rescale_bbox_scene.png)

```{.python continuation}
T.Rescale(keys="pos", method="bbox")
```

### method="linear"

Centers on the centroid and divides by the longest axis extent: the longest axis spans length 1.

=== "Object"

    ![Rescale method=linear on an object](../assets/transforms/rescale_linear.png)

=== "Scene"

    ![Rescale method=linear on a room](../assets/transforms/rescale_linear_scene.png)

```{.python continuation}
T.Rescale(keys="pos", method="linear")
```

## Scale

Multiplies the listed keys by a fixed factor.

=== "Object"

    ![Scale on an object](../assets/transforms/scale.png)

=== "Scene"

    ![Scale on a room](../assets/transforms/scale_scene.png)

```{.python continuation}
T.Scale(keys="pos", scale=0.5)
```

## Divide

Divides the listed keys by a fixed divisor, the counterpart of `Scale`. Both accept a single value broadcast to every key, or one value per key. The wireframe box is the bounding box and the `L` marker its vertical extent, which halves at `divisor=2`; the cloud also moves toward the origin, since dividing scales about the origin and not about the centroid.

=== "Object"

    ![Divide on an object](../assets/transforms/divide.png)

=== "Scene"

    ![Divide on a room](../assets/transforms/divide_scene.png)

```{.python continuation}
T.Divide(keys="pos", divisor=2.0)
```

`Divide` is a fixed divisor, unlike [`DivideKey`](utilities.md#dividekey), which divides by a tensor held under another key.

## Normalize

Per-channel standardization $x' = (x - \mu) / \max(\sigma, \epsilon)$, with `mean` and `std` broadcast against the last dimension. The usual target is `color` with dataset statistics, which is what the figure shows.

=== "Scene"

    ![Normalize applied to the color key](../assets/transforms/normalize.png)

```{.python continuation}
T.Normalize(keys="color", mean=(0.44, 0.41, 0.34), std=(0.22, 0.21, 0.20))
```

!!! note
    Standardized values leave $[0, 1]$, so a normalized `color` is no longer a displayable RGB triplet; the figure clips it back into range to render it.

## SubtractKey

Subtracts the tensor at `sub_keys` from the tensor at `keys` element-wise, so the offset lives in the data instead of the transform's arguments: block pipelines store a per-block center and subtract it here. `axes` restricts the subtraction to a subset of the last-dim components (XY only, keeping Z absolute). The arrow is the `center` vector; the dotted box is where the cloud used to sit.

=== "Object"

    ![SubtractKey on an object](../assets/transforms/subtract_key.png)

=== "Scene"

    ![SubtractKey on a room](../assets/transforms/subtract_key_scene.png)

```{.python continuation}
T.SubtractKey(keys="pos", sub_keys="center")
```

## BBoxCenter

Reads a flat bbox of shape $(2D,)$ laid out as $[\min_0, \ldots, \min_{D-1}, \max_0, \ldots, \max_{D-1}]$ and writes its per-axis midpoint $(\min + \max) / 2$ of shape $(D,)$. The two marked corners are `min` and `max`; the dot on the diagonal is the center they define. The usual pairing is `BBoxCenter` then `SubtractKey`, to recenter a block on the bbox a dataset stored with it.

=== "Object"

    ![BBoxCenter on an object](../assets/transforms/bbox_center.png)

=== "Scene"

    ![BBoxCenter on a room](../assets/transforms/bbox_center_scene.png)

```{.python continuation}
T.BBoxCenter(keys="block_bbox", dst_keys="block_center")
```

## Abs

Element-wise absolute value, folding every axis onto its positive side. The wireframe box is the bounding box and the gray dot the origin: after the fold the whole cloud sits in the positive octant.

=== "Object"

    ![Abs on an object](../assets/transforms/abs.png)

=== "Scene"

    ![Abs on a room](../assets/transforms/abs_scene.png)

```{.python continuation}
T.Abs(keys="pos")
```

## Rotation axes

Rotation is always around a single coordinate axis (`0`=X, `1`=Y, `2`=Z); Z is the gravity axis in the indoor convention. These images show the same rotation applied around each axis.

=== "Object"

    ![Rotation around each axis on an object](../assets/transforms/rotate_axes.png)

=== "Scene"

    ![Rotation around each axis on a room](../assets/transforms/rotate_axes_scene.png)

```{.python continuation}
import math

import torch
import torch_pointcloud.transforms.functional as F

pos = torch.randn(2048, 3)
R = F.rotation_matrix(math.radians(45.0), axis=2)  # 45 degrees around Z
pos = F.rotate_vectors(pos, R)
```

For random rotations in a training pipeline, use [`RandomRotate`](augmentation.md#randomrotate).

## Flip axes

Flipping negates one coordinate, mirroring the cloud across the plane orthogonal to that axis.

=== "Object"

    ![Flip across each axis on an object](../assets/transforms/flip_axes.png)

=== "Scene"

    ![Flip across each axis on a room](../assets/transforms/flip_axes_scene.png)

```{.python continuation}
pos = F.flip_vectors(pos, axis=0)  # mirror across X
```

For random flips in a training pipeline, use [`RandomFlip`](augmentation.md#randomflip).
