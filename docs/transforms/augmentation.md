# Augmentation

Augmentation decides what the model is invariant to. Every transform here draws its own randomness, takes a probability `p`, and accepts a `generator` for reproducible runs.

## Which one to use

| You want the model invariant to        | Use                                                      |
| --------------------------------------- | -------------------------------------------------------- |
| Which way the object faces              | `RandomRotate`, `RandomRotateChoice`                     |
| Handedness (left / right chirality)     | `RandomFlip`                                             |
| Sensor noise on the coordinates         | `RandomJitter`                                           |
| Object or room size                     | `RandomScale`                                            |
| Where the cloud sits                    | `RandomShift`                                            |
| Scan density and occlusion              | `RandomDropout`                                          |
| Small non-rigid deformation             | `RandomElasticDistortion`                                |
| Lighting and camera response            | [Color transforms](color.md)                             |
| Objects appearing out of context        | [`Mix3D`, `LaserMix`, `PolarMix`](#mix-two-scenes)       |

Parameters for each are in the [API reference](../api/transforms/transforms.md).

## Rotate

`RandomRotate` draws a uniform angle in degrees from `angle_range` and applies it to every listed key, so `normal` turns with `pos`.

=== "Object"

    ![RandomRotate on an object](../assets/transforms/random_rotate.png)

=== "Scene"

    ![RandomRotate on a room](../assets/transforms/random_rotate_scene.png)

```python
import torch_pointcloud.transforms as T

T.RandomRotate(keys=("pos", "normal"), angle_range=(-180.0, 180.0), axis=2)
```

Full rotation about the gravity axis (`axis=2`) is safe indoors and outdoors. Rotating about X or Y tips the scene over, so keep those to a few degrees.

`RandomRotateChoice` draws from a discrete list instead of a continuous range, the usual ModelNet and ScanObjectNN setting:

=== "Object"

    ![RandomRotateChoice on an object](../assets/transforms/random_rotate_choice.png)

=== "Scene"

    ![RandomRotateChoice on a room](../assets/transforms/random_rotate_choice_scene.png)

```{.python continuation}
T.RandomRotateChoice(keys="pos", angles=[0.0, 90.0, 180.0, 270.0], axis=2)
```

See [Rotate or flip by hand](geometric.md#rotate-or-flip-by-hand) for the axis convention.

## Flip

`RandomFlip` mirrors across each axis in `axes`, sampling once per call so every key stays consistent.

=== "Object"

    ![RandomFlip on an object](../assets/transforms/random_flip.png)

=== "Scene"

    ![RandomFlip on a room](../assets/transforms/random_flip_scene.png)

```{.python continuation}
T.RandomFlip(keys="pos", axes=(0, 1), p=0.5)
```

Flip the horizontal axes freely. Flipping the gravity axis puts the ceiling on the floor, which no test scene does.

## Perturb the geometry

`RandomJitter` adds clipped Gaussian noise per point, in place of sensor noise. `clip` bounds a rare large draw, which would otherwise move a point far from the surface.

=== "Object"

    ![RandomJitter on an object](../assets/transforms/random_jitter.png)

=== "Scene"

    ![RandomJitter on a room](../assets/transforms/random_jitter_scene.png)

```{.python continuation}
T.RandomJitter(keys="pos", sigma=0.01, clip=0.05)
```

`RandomScale` covers size variation, with one global factor or per-axis when `anisotropic=True`.

=== "Object"

    ![RandomScale on an object](../assets/transforms/random_scale.png)

=== "Scene"

    ![RandomScale on a room](../assets/transforms/random_scale_scene.png)

```{.python continuation}
T.RandomScale(keys="pos", scale_range=(0.8, 1.25))
```

`RandomShift` translates by one random vector shared across every listed key.

=== "Object"

    ![RandomShift on an object](../assets/transforms/random_shift.png)

=== "Scene"

    ![RandomShift on a room](../assets/transforms/random_shift_scene.png)

```{.python continuation}
T.RandomShift(keys="pos", shift_range=(-0.2, 0.2))
```

!!! warning "Scale and shift change what a voxel size means"
    If a `Voxelize` or `Quantize` follows, these two move points across cell boundaries. Scaling by 2 with a fixed 2 cm grid is the same as scaling by 1 with a 1 cm grid, so tune the two together.

## Vary the density

`RandomDropout` drops a fraction of the points, with one shared keep-mask across every listed key. It stands in for occlusion and for a sparser sensor than the training one.

=== "Object"

    ![RandomDropout on an object](../assets/transforms/random_dropout.png)

=== "Scene"

    ![RandomDropout on a room](../assets/transforms/random_dropout_scene.png)

```{.python continuation}
T.RandomDropout(keys=("pos", "color"), p_drop=0.1)
```

## Deform

`RandomElasticDistortion` warps the coordinates with a smooth random displacement field, the SparseConvNet indoor recipe. Compose two at different `granularity` to get coarse and fine deformation at once.

=== "Object"

    ![RandomElasticDistortion on an object](../assets/transforms/random_elastic_distortion.png)

=== "Scene"

    ![RandomElasticDistortion on a room](../assets/transforms/random_elastic_distortion_scene.png)

```{.python continuation}
T.Compose([
    T.RandomElasticDistortion(keys="pos", granularity=0.2, magnitude=0.4),
    T.RandomElasticDistortion(keys="pos", granularity=0.8, magnitude=1.6),
])
```

## Mix two scenes

The three mixes are the only transforms here that read two samples. They are called as `mix(data, other)` and cannot go inside a `Compose`. `MixDataset` draws the second sample at a random index:

```{.python notest}
from torch_pointcloud.datasets import MixDataset

train = MixDataset(
    train_dataset,
    mix=T.Mix3D(keys=("pos", "color", "segment"), instance_key="instance"),
)
```

Each mix takes its own `p`, the probability of merging. Below `p`, the first sample comes back unchanged. In the figures below, the first two panels are the inputs and the third is the result, colored by the sample each point came from.

### Mix3D

The out-of-context mix of :arxiv: [Mix3D](https://arxiv.org/abs/2110.02210). Both scenes are concatenated along the point dimension, so the result holds roughly twice as many points as either input, and objects appear in rooms they were never scanned in. With `instance_key` present in both, the second scene's instance ids are shifted past the first's maximum so the merged instances stay disjoint; points labeled `ignore_index` keep that label.

![Mix3D on two ScanNet rooms](../assets/transforms/mix3d.png)

```{.python continuation}
T.Mix3D(keys=("pos", "color", "segment"), instance_key="instance")
```

### LaserMix

The LiDAR mix of :arxiv: [LaserMix](https://arxiv.org/abs/2207.00026). Both scans are split into `num_areas` inclination bands, and alternating bands come from each scan, so the result still covers the full field of view. One band count is drawn per call, and every key is masked with the same selection.

![LaserMix on two LiDAR scans](../assets/transforms/laser_mix.png)

```{.python continuation}
T.LaserMix(keys=("pos", "segment"), num_areas=(3, 4, 5, 6), pitch_range=(-25.0, 3.0))
```

### PolarMix

The LiDAR mix of :arxiv: [PolarMix](https://arxiv.org/abs/2208.00223), which runs two independent sub-augmentations. With probability `swap_ratio` a random azimuth half-sector of the first scan is replaced by the same sector of the second. With probability `rotate_paste_ratio` the second scan's points whose label is in `instance_classes` are rotated by a random angle about the up axis and appended; only `pos_key` is rotated for those points, and the other keys are copied as they are.

![PolarMix on two LiDAR scans](../assets/transforms/polar_mix.png)

```{.python continuation}
T.PolarMix(keys=("pos", "segment"), instance_classes=(1, 2, 3), swap_ratio=0.5)
```
