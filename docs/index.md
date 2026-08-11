# torch-pointcloud

A PyTorch library for deep learning on point clouds. Production-ready models for classification, segmentation, and detection, with a `create_model` factory, pretrained-weight registry, and composable transforms in the style of :pytorch: [`timm`](https://github.com/huggingface/pytorch-image-models) and :pyg: [`torch_geometric`](https://pytorch-geometric.readthedocs.io/).

```{.python notest}
import torch
import torch_pointcloud as tp
import torch_pointcloud.transforms as T

# 1. Load a pretrained model
model = tp.create_model("pointnext-sm.scanobjectnn.openpoints", task="classification", pretrained=True).eval()

# 2. Build a transform pipeline
transform = T.Compose([
    T.Rescale(keys="pos", method="centroid"),
    T.RandomSample(keys=("pos", "color"), num_samples=1024),
])

# 3. Run inference
data = transform({"pos": torch.randn(2048, 3), "color": torch.rand(2048, 3)})
logits = model(data["color"], data["pos"], batch=torch.zeros(1024, dtype=torch.long))
```

## What's inside

<div class="grid cards" markdown>

-   :material-rocket-launch: __[Get Started](get-started.md)__

    Install, run your first model, and learn the library's conventions in fifteen lines.

-   :material-cube-outline: __[Models](models/overview.md)__

    PointNet, PointNet++, DGCNN, KPConv, PointNeXt, OctFormer, Point Transformer V1-V3, PVCNN, SPVCNN, RandLA-Net, and more. All registered with a timm-style factory.

-   :material-database: __[Datasets](datasets/overview.md)__

    ModelNet, ScanNet, S3DIS, ShapeNetPart, ScanObjectNN, SemanticKITTI, Semantic3D, Toronto3D, ParisLille3D.

-   :material-tune: __[Transforms](transforms/overview.md)__

    Composable, non-mutating dict transforms inspired by [MONAI](https://docs.monai.io/): `Compose`, `Rescale`, `Shift`, `Voxelize`, `FarthestPointSample`, mask family, and more.

-   :material-book-open-page-variant: __[API Reference](api/index.md)__

    Auto-generated reference for every public class and function.

-   :material-github: __[Source](https://github.com/arthurdjn/pytorch-pointcloud)__

    Browse the source, file issues, or contribute.

</div>

## Design principles

1. **Packed (PyG-style) batches.** All point clouds use `pos: (N, 3)` + `batch: (N,)` rather than padded `(B, N, 3)` tensors.
2. **Composable transforms.** Single-purpose, non-mutating dict transforms chained via `Compose`. Every transform has a tensor-level functional equivalent under `torch_pointcloud.transforms.functional`.
3. **Pretrained weights as a registry.** `create_model(name, pretrained=True)` looks up the weights and the matching transform pipeline.
4. **Type-safe.** Every public function has a fully-typed signature; `Literal` types are exposed as named aliases (`ShiftMethod`, `RescaleMethod`, `ReduceOp`).

## License

MIT. See [`LICENSE`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/LICENSE).
