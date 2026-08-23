# SimpleInferer

`SimpleInferer` calls the predictor on the whole scene in one forward pass: no tiling, no blending.

```{.python notest}
import torch_pointcloud as tp
from torch_pointcloud.inferers import SimpleInferer

model = tp.create_model("pointnet2.s3dis-area5.xu-yan", task="segmentation", pretrained=True).eval()

inferer = SimpleInferer()
logits = inferer(scene, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
```

## When to use

Objects and small scenes that fit in memory at native resolution, and as the base inside [`TTAInferer`](tta.md). If a full room runs out of memory, switch to [`SlidingWindowInferer`](sliding-window.md) or [`KNNWindowInferer`](knn-window.md); the calling code does not change.
