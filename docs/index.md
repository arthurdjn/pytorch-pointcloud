# PyTorch PointCloud

![PyTorch-PointCloud](./assets/pytorch-pointcloud.png)

A PyTorch library for deep learning on point clouds. Production-ready models for classification, segmentation, and detection, with a `create_model` factory, pretrained-weight registry, and composable transforms in the style of :pytorch: [`timm`](https://github.com/huggingface/pytorch-image-models) and :pyg: [`torch_geometric`](https://pytorch-geometric.readthedocs.io/).

<div class="tp-tasks" markdown>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/classification.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A chair turning a full circle, in the accent color, classified by a pretrained PointNet++"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/classification_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A chair turning a full circle, in the accent color, classified by a pretrained PointNet++"></video>
<figcaption markdown="span">**Classify objects**`pointnet2-ssg.modelnet40.xu-yan`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/part_segmentation.mp4" autoplay loop muted playsinline preload="metadata" aria-label="An airplane turning a full circle, its wings, body, tail and engines each in their own color"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/part_segmentation_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="An airplane turning a full circle, its wings, body, tail and engines each in their own color"></video>
<figcaption markdown="span">**Segment parts**`pointnext-sm.shapenetpart.openpoints`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/indoor.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A camera gliding from room to room over a scanned house sliced open above the furniture, crossing in turn from its true color, to the semantic class predicted for every point, to a wireframe box around each piece of furniture the instance head found"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/indoor_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A camera gliding from room to room over a scanned house sliced open above the furniture, crossing in turn from its true color, to the semantic class predicted for every point, to a wireframe box around each piece of furniture the instance head found"></video>
<figcaption markdown="span">**Understand a scene**`ptv3-base.scannet20.pointcept`<br>`oneformer3d-base.scannet20.danila-rukhovich`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/driving.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A bird's-eye camera riding above the car down a LiDAR sequence, one sweep per frame, every point colored by its predicted class and every vehicle boxed as it goes past"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/driving_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A bird's-eye camera riding above the car down a LiDAR sequence, one sweep per frame, every point colored by its predicted class and every vehicle boxed as it goes past"></video>
<figcaption markdown="span">**Drive**`spvcnn-119gmacs.semantickitti.mit-han-lab`<br>`second.kitti.openpcdet`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/survey.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A slow turn around the Eiffel Tower as an airborne survey recorded it, crossing from the sensor's own return strength, to the classification the survey ships with, to the principal components an encoder reads off the geometry alone"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/survey_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A slow turn around the Eiffel Tower as an airborne survey recorded it, crossing from the sensor's own return strength, to the classification the survey ships with, to the principal components an encoder reads off the geometry alone"></video>
<figcaption markdown="span">**Embed a survey**`utonia-lp.scannet20.pointcept`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/similarity.mp4" autoplay loop muted playsinline preload="metadata" aria-label="The same house seen by a self-supervised encoder: first colored by the principal components of its features, then queried one object at a time, so that asking from a single chair lights every chair in the house, and asking from a table or a sofa lights those instead"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/similarity_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="The same house seen by a self-supervised encoder: first colored by the principal components of its features, then queried one object at a time, so that asking from a single chair lights every chair in the house, and asking from a table or a sofa lights those instead"></video>
<figcaption markdown="span">**Search by similarity**`sonata-lp.scannet20.fair`</figcaption>
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
