# SimpleInferer

`SimpleInferer` calls the predictor on the whole scene in one forward pass: no tiling, no blending. It is the lightest possible [`Inferer`](overview.md) and the baseline every other strategy is measured against.

![One predictor call over the whole scene](../assets/animations/simple.webp)

One call, every point, nothing to stitch: the baseline the other strategies are measured against.

## Usage

```{.python notest}
import torch_pointcloud as tp
from torch_pointcloud.inferers import SimpleInferer

model = tp.create_model(
    "pointnet2.s3dis-area5.xu-yan", task="segmentation", pretrained=True
).eval()

inferer = SimpleInferer()
logits = inferer(
    scene, predictor=lambda d: model(d["x"], d["pos"], d["batch"])
)
```

## When to use

- **Object classification** and part segmentation, where each cloud is a few thousand points.
- **Small scenes** that fit GPU memory at native resolution in one pass.
- As the **base inferer** inside [`TTAInferer`](tta.md) when the scene is small but augmentation averaging is still wanted.
- If a full room runs out of memory, switch to [`SlidingWindowInferer`](sliding-window.md) or [`KNNWindowInferer`](knn-window.md); the calling code does not change.
