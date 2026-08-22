# Color

Color transforms act on the `color` key, shown on the ScanNet room with its true per-point RGB (room only: the object has no colors). Strengths are exaggerated so the effect is visible. Default range is $[0, 1]$; set `int_color=True` for $[0, 255]$ colors.

| Transform                                                                                                                   | Purpose                                      |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [`RandomColorJitter`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomColorJitter)             | Perturb brightness, contrast, and saturation |
| [`RandomColorShift`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomColorShift)               | Add a random per-channel color offset        |
| [`RandomColorGrayScale`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomColorGrayScale)       | Convert colors to grayscale                  |
| [`RandomColorDrop`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomColorDrop)                 | Replace colors with a constant fill          |
| [`RandomColorAutoContrast`](../api/transforms/transforms.md#torch_pointcloud.transforms.transforms.RandomColorAutoContrast) | Stretch the color range to full contrast     |

## RandomColorJitter

Perturbs brightness, contrast, and saturation by random strengths.

=== "Scene"

    ![Perturbs brightness, contrast, saturation](../assets/transforms/color_jitter.png)

```python
import torch_pointcloud.transforms as T

T.RandomColorJitter(keys="color", brightness=0.4, contrast=0.4, saturation=0.2)
```

## RandomColorShift

Adds one uniform offset per channel, clamped to the valid range.

=== "Scene"

    ![Adds a random color offset](../assets/transforms/color_shift.png)

```{.python continuation}
T.RandomColorShift(keys="color", shift_range=(-0.05, 0.05))
```

## RandomColorGrayScale

Converts to BT.601 luminance with probability `p`.

=== "Scene"

    ![Converts colors to gray](../assets/transforms/color_grayscale.png)

```{.python continuation}
T.RandomColorGrayScale(keys="color", p=0.2)
```

## RandomColorDrop

Replaces all colors with a constant gray fill with probability `p`.

=== "Scene"

    ![Replaces colors with a constant](../assets/transforms/color_drop.png)

```{.python continuation}
T.RandomColorDrop(keys="color", fill=0.5, p=0.2)
```

## RandomColorAutoContrast

Stretches the per-cloud color range to full extent, blended with the input by `blend`.

=== "Scene"

    ![Stretches the color range](../assets/transforms/color_auto_contrast.png)

```{.python continuation}
T.RandomColorAutoContrast(keys="color", blend=0.5, p=0.2)
```
