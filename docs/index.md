# PyTorch PointCloud

![PyTorch-PointCloud](./assets/pytorch-pointcloud.png)

A PyTorch library for deep learning on point clouds. Models for classification, segmentation, and detection, pretrained-weight registry, and composable transforms in the style of :pytorch: [`timm`](https://github.com/huggingface/pytorch-image-models) and :pyg: [`torch_geometric`](https://pytorch-geometric.readthedocs.io/).

<div class="tp-tasks" markdown>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/classification.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A chair turning a full circle, in the accent color, classified by a pretrained PointNet++"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/classification_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A chair turning a full circle, in the accent color, classified by a pretrained PointNet++"></video>
<figcaption markdown="span">**Object classification**`pointnet2-ssg.modelnet40.xu-yan`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/part_segmentation.mp4" autoplay loop muted playsinline preload="metadata" aria-label="An airplane turning a full circle, its wings, body, tail and engines each in their own color"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/part_segmentation_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="An airplane turning a full circle, its wings, body, tail and engines each in their own color"></video>
<figcaption markdown="span">**Part segmentation**`pointnext-sm.shapenetpart.openpoints`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/indoor.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A camera gliding from room to room over a scanned house sliced open above the furniture, crossing in turn from its true color, to the semantic class predicted for every point, to a wireframe box around each piece of furniture the instance head found"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/indoor_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A camera gliding from room to room over a scanned house sliced open above the furniture, crossing in turn from its true color, to the semantic class predicted for every point, to a wireframe box around each piece of furniture the instance head found"></video>
<figcaption markdown="span">**Indoor segmentation / detection**`ptv3-base.scannet20.pointcept`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/driving.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A bird's-eye camera riding above the car down a LiDAR sequence, one sweep per frame, every point colored by its predicted class and every vehicle boxed as it goes past"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/driving_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A bird's-eye camera riding above the car down a LiDAR sequence, one sweep per frame, every point colored by its predicted class and every vehicle boxed as it goes past"></video>
<figcaption markdown="span">**Outdoor segmentation / detection**`spvcnn-119gmacs.semantickitti.mit-han-lab`<br>`second.kitti.openpcdet`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/survey.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A slow turn around the Eiffel Tower as an airborne survey recorded it, crossing from the sensor's own return strength, to the classification the survey ships with, to the principal components an encoder reads off the geometry alone"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/survey_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="A slow turn around the Eiffel Tower as an airborne survey recorded it, crossing from the sensor's own return strength, to the classification the survey ships with, to the principal components an encoder reads off the geometry alone"></video>
<figcaption markdown="span">**Large scale segmentation**`utonia-lp.scannet20.pointcept`</figcaption>
</figure>

<figure markdown="1">
<video class="tp-tile tp-tile--light" src="./assets/animations/hero/similarity.mp4" autoplay loop muted playsinline preload="metadata" aria-label="The same house seen by a self-supervised encoder: first colored by the principal components of its features, then queried one object at a time, so that asking from a single chair lights every chair in the house, and asking from a table or a sofa lights those instead"></video>
<video class="tp-tile tp-tile--dark" src="./assets/animations/hero/similarity_dark.mp4" autoplay loop muted playsinline preload="metadata" aria-label="The same house seen by a self-supervised encoder: first colored by the principal components of its features, then queried one object at a time, so that asking from a single chair lights every chair in the house, and asking from a table or a sofa lights those instead"></video>
<figcaption markdown="span">**Feature extraction**`sonata-lp.scannet20.fair`</figcaption>
</figure>

</div>

## Why torch-pointcloud?

`torch-pointcloud` is not a training framework: it packages point cloud models, their checkpoints, datasets and
transforms as a library you import in your own PyTorch code. It complements the codebases the checkpoints come from.

|          | torch-pointcloud                   | Pointcept                    | OpenPCDet, MMDetection3D     | PyG                    |
| -------- | ---------------------------------- | ---------------------------- | ---------------------------- | ---------------------- |
| Kind     | Library                            | Research codebase            | Detection toolboxes          | Graph learning library |
| Workflow | Import it in your own PyTorch code | Configs and training scripts | Configs and training scripts | Build your own models  |
| Weights  | Ported from the original codebases | Its own checkpoints          | Its own checkpoints          | None for point clouds  |

## What's inside

<div class="grid cards" markdown>

-   :material-rocket-launch: __[Get Started](get-started.md)__

    Install, run your first model, and learn the library's conventions.

-   :material-cube-outline: __[Models](models/overview.md)__

    PointNet, PointNet++, RandLA-Net, KPConv, PointNeXt, OctFormer, Point Transformer, SPVCNN, and more.

-   :material-database: __[Datasets](datasets/overview.md)__

    ModelNet, ScanNet, S3DIS, ShapeNetPart, ScanObjectNN, SemanticKITTI, Semantic3D, and more.

-   :material-tune: __[Transforms](transforms/overview.md)__

    Composable, non-mutating dict transforms inspired by :monai: [MONAI](https://docs.monai.io/).

-   :material-school: __[Tutorials](examples/index.md)__

    Ready-to-use notebooks, from a first classification to survey-scale inference.

-   :material-book-open-page-variant: __[API Reference](api/index.md)__

    Auto-generated reference for every public class and function.

-   :material-github: __[Source](https://github.com/arthurdjn/pytorch-pointcloud)__

    Browse the source, file issues, or contribute.

</div>

## License

Apache 2.0, see [`LICENSE`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/LICENSE). Pretrained weights keep the
license of their source, see [`THIRD_PARTY_NOTICES.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/THIRD_PARTY_NOTICES.md).
