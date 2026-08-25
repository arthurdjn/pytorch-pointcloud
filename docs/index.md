# PyTorch PointCloud

![PyTorch-PointCloud](./assets/pytorch-pointcloud.png)

A PyTorch library for deep learning on point clouds. Production-ready models for classification, segmentation, and detection, with a `create_model` factory, pretrained-weight registry, and composable transforms in the style of :pytorch: [`timm`](https://github.com/huggingface/pytorch-image-models) and :pyg: [`torch_geometric`](https://pytorch-geometric.readthedocs.io/).

<div class="tp-tasks" markdown>

<figure markdown="1">
<video src="./assets/animations/hero/classification.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A chair turning a full circle, in the accent color, classified by a pretrained PointNet++"></video>
<figcaption markdown="span">**Classify objects**`pointnet2-ssg.modelnet40.xu-yan`</figcaption>
</figure>

<figure markdown="1">
<video src="./assets/animations/hero/part_segmentation.mp4" autoplay loop muted playsinline preload="metadata" aria-label="An airplane turning a full circle, its wings, body, tail and engines each in their own color"></video>
<figcaption markdown="span">**Segment parts**`pointnext-sm.shapenetpart.openpoints`</figcaption>
</figure>

<figure markdown="1">
<video src="./assets/animations/hero/scene_segmentation.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A camera gliding over a scanned wing of offices and a corridor with the ceiling cut away, its true color crossing over to the semantic class predicted for every point"></video>
<figcaption markdown="span">**Segment scenes**`ptv3-base.s3dis-area5.pointcept`</figcaption>
</figure>

<figure markdown="1">
<video src="./assets/animations/hero/detection.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A camera gliding over the same wing, with a wireframe box around each chair, table, bookcase and board the instance head found"></video>
<figcaption markdown="span">**Detect objects**`oneformer3d-base.s3dis-area5.danila-rukhovich`</figcaption>
</figure>

<figure markdown="1">
<video src="./assets/animations/hero/driving.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A bird's-eye camera riding above the car down a LiDAR sequence, one sweep per frame, every point colored by its predicted class and every vehicle boxed as it goes past"></video>
<figcaption markdown="span">**Drive**`spvcnn-119gmacs.semantickitti.mit-han-lab`<br>`second.kitti.openpcdet`</figcaption>
</figure>

<figure markdown="1">
<video src="./assets/animations/hero/features.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A camera circling a scanned room colored by the principal components of a self-supervised encoder's per-point features"></video>
<figcaption markdown="span">**Embed points**`sonata-lp.scannet20.fair`</figcaption>
</figure>

</div>

## In a few lines

```{.python notest}
import numpy as np
import torch
from plyfile import PlyData

import torch_pointcloud as tp
from torch_pointcloud.utils.data import collate

# Load pretrained checkpoint and sample cloud.
model, info = tp.create_model(
    "pointnet2-ssg.modelnet40.xu-yan",
    task="classification",
    pretrained=True,
    return_info=True,
)
model = model.eval()

# Get associated transform pipeline.
transform = info["transform"]

# Preprocess the input.
ply = PlyData.read("sample.ply")["vertex"]
pos = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype("float32")
sample = {"pos": torch.from_numpy(pos)}
sample = transform(sample)

# Preprocess, pack into a batch, predict.
batch = collate([sample])
with torch.no_grad():
    logits = model(None, batch["pos"], batch["batch"])

print(f"Prediction: {logits.argmax().item()}")
# Prediction: 0
```

## What's inside

<div class="grid cards" markdown>

-   :material-rocket-launch: __[Get Started](get-started.md)__

    Install, run your first model, and learn the library's conventions in fifteen lines.

-   :material-cube-outline: __[Models](models/overview.md)__

    PointNet, PointNet++, RandLA-Net, KPConv, PointNeXt, OctFormer, Point Transformer, SPVCNN, and more.

-   :material-database: __[Datasets](datasets/overview.md)__

    ModelNet, ScanNet, S3DIS, ShapeNetPart, ScanObjectNN, SemanticKITTI, Semantic3D, and more.

-   :material-tune: __[Transforms](transforms/overview.md)__

    Composable, non-mutating dict transforms inspired by :monai: [MONAI](https://docs.monai.io/).

-   :material-school: __[Tutorials](examples/index.md)__

    Ready to use notebooks, from a first classification to survey-scale inference.

-   :material-book-open-page-variant: __[API Reference](api/index.md)__

    Auto-generated reference for every public class and function.

-   :material-github: __[Source](https://github.com/arthurdjn/pytorch-pointcloud)__

    Browse the source, file issues, or contribute.

</div>

## License

Apache 2.0. See [`LICENSE`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/LICENSE).
