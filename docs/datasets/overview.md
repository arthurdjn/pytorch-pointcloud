# Datasets

:pytorch-pointcloud-mini: `torch-pointcloud` provides several datasets for benchmarking and training. Each dataset returns a single `dict` (the format consumed by [transforms](../transforms/overview.md)) and integrates with `torch.utils.data.DataLoader` via the `collate` helper in `torch_pointcloud.utils.data`.

For example:

```{.python notest}
from torch.utils.data import DataLoader
from torch_pointcloud.datasets import ModelNet40
from torch_pointcloud.utils.data import collate

dataset = ModelNet40(root="data", train=True, download=True)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate)
```

`PointCloudDataLoader` is the same `DataLoader` with the `collate` helper already applied.

```{.python notest}
from torch_pointcloud.utils.data import PointCloudDataLoader
from torch_pointcloud.datasets import ModelNet40

dataset = ModelNet40(root="data", train=True, download=True)
dataloader = PointCloudDataLoader(dataset, batch_size=32)
```

## Tasks

<div class="grid cards" markdown>

-   :material-shape-outline: __[Classification](classification.md)__

    ModelNet and ScanObjectNN: meshes, presampled clouds, difficulty variants.

-   :material-floor-plan: __[Segmentation](segmentation.md)__

    Indoor rooms and outdoor LiDAR, their splits, and the label conventions.

-   :material-puzzle-outline: __[Part segmentation](part-segmentation.md)__

    ShapeNetPart: 16 categories, 50 global part ids.

-   :material-cube-scan: __[Detection](detection.md)__

    SUN RGB-D, KITTI, nuScenes, and batching ragged boxes.

</div>

### Object classification

| Dataset                                                    | Paper                                                                                                                                                  | Samples | Classes                  |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | ------------------------ |
| **[ModelNet10 / ModelNet40](../api/datasets/modelnet.md)** | :arxiv: [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://arxiv.org/abs/1406.5670)                                                   | ~12k    | 10 / 40                  |
| **[ModelNet40Hdf5](../api/datasets/modelnet.md)**          | :arxiv: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)                               | ~12k    | 40                       |
| **[ShapeNetPart](../api/datasets/shapenetpart.md)**        | :arxiv: [A Scalable Active Framework for Region Annotation in 3D Shape Collections](https://dl.acm.org/doi/10.1145/2980179.2980238)                    | ~16k    | 16 categories / 50 parts |
| **[ScanObjectNN](../api/datasets/scanobjectnn.md)**        | :arxiv: [Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data](https://arxiv.org/abs/1908.04616) | 2.9k    | 15                       |

### Indoor scene segmentation

| Dataset                                      | Paper                                                                                                                                                             | Scenes             | Classes          |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------- |
| **[S3DIS](../api/datasets/s3dis.md)**        | :arxiv: [3D Semantic Parsing of Large-Scale Indoor Spaces](https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf) | 271 rooms, 6 areas | 13               |
| **[ScanNet v2](../api/datasets/scannet.md)** | :arxiv: [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](https://arxiv.org/abs/1702.04405)                                                         | 1.5k scenes        | 20 (NYU40) / 200 |

### Outdoor / driving segmentation

| Dataset                                               | Paper                                                                                                                                                                     | Frames     | Classes |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- |
| **[SemanticKITTI](../api/datasets/semantickitti.md)** | :arxiv: [SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences](https://arxiv.org/abs/1904.01416)                                                  | 43k frames | 19      |
| **[Semantic3D](../api/datasets/semantic3d.md)**       | :arxiv: [Semantic3D.net: A new Large-scale Point Cloud Classification Benchmark](https://arxiv.org/abs/1704.03847)                                                        | 30 scenes  | 8       |
| **[Toronto3D](../api/datasets/toronto3d.md)**         | :arxiv: [Toronto-3D: A Large-scale Mobile LiDAR Dataset for Semantic Segmentation of Urban Roadways](https://arxiv.org/abs/2003.08284)                                    | 4 areas    | 8       |
| **[ParisLille3D](../api/datasets/parislille3d.md)**   | :arxiv: [Paris-Lille-3D: a large and high-quality ground truth urban point cloud dataset for automatic segmentation and classification](https://arxiv.org/abs/1712.00032) | 3 scenes   | 9       |

### Base class

| Dataset                                                | Task  | Notes                                                                                                                                   |
| ------------------------------------------------------ | ----- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **[PointCloudDataset](../api/datasets/pointcloud.md)** | (any) | Abstract base class all loaders build on: `raw/` + `processed/` disk layout, `download` / `process` hooks. Subclass it for custom data. |

## About dict keys

All datasets emit dicts using the standard key conventions from `DataKeys` in `torch_pointcloud.utils.data`:

| Key        | Shape    | Description                                 |
| ---------- | -------- | ------------------------------------------- |
| `pos`      | $(N, 3)$ | 3D coordinates                              |
| `color`    | $(N, 3)$ | RGB (uint8 or float, depending on dataset)  |
| `normal`   | $(N, 3)$ | Surface normals (when available)            |
| `segment`  | $(N,)$   | Semantic labels (segmentation datasets)     |
| `instance` | $(N,)$   | Instance IDs (ScanNet)                      |
| `label`    | scalar   | Object class (classification datasets)      |
| `face`     | $(F, 3)$ | Triangle indices (ModelNet / mesh datasets) |

After `collate`, per-point tensors are concatenated along axis 0 and a `batch` key of shape $(N,)$ gives each point's source scene.

!!! warning "Color conventions vary per dataset"
    `color` is uint8 in $[0, 255]$ for the raw-value loaders (`S3DIS`, `ScanNet`, `Toronto3D`, `Semantic3D`, `SunRGBD`) and float32 in $[0, 1]$ for `S3DISHdf5`, which ships pre-normalized values.

!!! warning "Ignore-index conventions vary per dataset"
    Unlabeled points use label 0 (`<unk>` / outdoor conventions, e.g. `ScanNet`), -1 (indoor no-instance and class-subset remaps, e.g. `S3DIS`), or 255 (the `SemanticKITTI` remap example).
