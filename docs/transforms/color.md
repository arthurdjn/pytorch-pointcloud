# Color

An indoor checkpoint reads RGB, and RGB is the least stable thing a scan carries: the same room photographs differently under daylight and a ceiling lamp, and two sensors disagree on white balance. Color augmentations are how you stop the model from keying on that. They act on the `color` key and leave the geometry alone.

## Which one to reach for

| You want the model invariant to     | Reach for                  |
| ------------------------------------ | -------------------------- |
| Lighting and camera response         | `RandomColorJitter`        |
| A global color cast                  | `RandomColorShift`         |
| Color being available at all         | `RandomColorGrayScale`     |
| Color being available at all, harder | `RandomColorDrop`          |
| A washed-out or over-saturated scan  | `RandomColorAutoContrast`  |

Parameters for each are in the [API reference](../api/transforms/transforms.md); this page is about which to pick.

!!! warning "Check the range first"
    These default to colors in $[0, 1]$. `S3DIS`, `ScanNet`, `Toronto3D` and `Semantic3D` hand you uint8 in $[0, 255]$, so pass `int_color=True` or divide first, or a `shift_range` of $\pm 0.05$ becomes a change you cannot see. See [Segmentation datasets](../datasets/segmentation.md) for which loader is which.

The figures below use the ScanNet room with its true per-point RGB, at exaggerated strengths so the effect is visible. The object samples carry no color, so there is no Object tab on this page.

## Vary lighting and white balance

`RandomColorJitter` perturbs brightness, contrast and saturation by independent random strengths, which is the workhorse and usually the only one you need.

![Perturbs brightness, contrast, saturation](../assets/transforms/color_jitter.png)

```python
import torch_pointcloud.transforms as T

T.RandomColorJitter(keys="color", brightness=0.4, contrast=0.4, saturation=0.2)
```

`RandomColorShift` adds one uniform offset per channel, clamped back into range: a global cast rather than a per-point perturbation.

![Adds a random color offset](../assets/transforms/color_shift.png)

```{.python continuation}
T.RandomColorShift(keys="color", shift_range=(-0.05, 0.05))
```

## Force the model off color

If the model leans on color it will fail on a scan that has none. `RandomColorGrayScale` converts to BT.601 luminance with probability `p`, keeping the shading; `RandomColorDrop` is the blunter version, replacing every color with a constant fill so the sample carries geometry only.

![Converts colors to gray](../assets/transforms/color_grayscale.png)

```{.python continuation}
T.RandomColorGrayScale(keys="color", p=0.2)
```

![Replaces colors with a constant](../assets/transforms/color_drop.png)

```{.python continuation}
T.RandomColorDrop(keys="color", fill=0.5, p=0.2)
```

Keep `p` low. These are meant to make color a helpful signal rather than a load-bearing one, and at a high `p` you are simply training a geometry-only model on a fraction of your data.

## Normalize the exposure

`RandomColorAutoContrast` stretches the per-cloud color range to its full extent, blended with the input by `blend`. It pulls a dim or washed-out scan toward the same range as the rest of the set, so it evens out the scans rather than perturbing them.

![Stretches the color range](../assets/transforms/color_auto_contrast.png)

```{.python continuation}
T.RandomColorAutoContrast(keys="color", blend=0.5, p=0.2)
```

Apply the color augmentations before any [`Normalize`](geometric.md#standardize-colors): once colors are standardized they are no longer in a color range, and a brightness multiply on a standardized value does not mean what you want it to.
