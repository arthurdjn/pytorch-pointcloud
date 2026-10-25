# Datasets

:pytorch-pointcloud-mini: `torch-pointcloud` provides loaders for the common point cloud benchmarks. Each sample is one `dict`, the format
[transforms](../transforms/overview.md) consume.

Datasets with an automatic download take `download=True`; the others must be downloaded manually. S3DIS, ScanNet and
ScanObjectNN are released under terms of use: read them (each class links its `terms_url`), then pass
`accept_terms=True` next to `download=True`. The original files are kept in `root/<dataset>/raw`, and the cache built
from them in `processed`.

```{.python notest}
from torch_pointcloud.datasets import ModelNet40
from torch_pointcloud.utils.data import PointCloudDataLoader

dataset = ModelNet40(root="data", train=True, download=True)
dataloader = PointCloudDataLoader(dataset, batch_size=32)
```

`PointCloudDataLoader` is a `DataLoader` using the packed-batch `collate` of `torch_pointcloud.utils.data`.

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

-   :material-vector-combine: __[Registration](registration.md)__

    3DMatch and 3DLoMatch fragment pairs with their ground-truth transforms.

</div>

### Object classification

<div class="tp-index" markdown>

| Dataset | Samples | Classes |
| --- | ---: | ---: |
| [ModelNet10 / ModelNet40](../api/datasets/modelnet.md)<br><span class="tp-index__detail">:arxiv: [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://arxiv.org/abs/1406.5670)</span> | ~12k | 10 / 40 |
| [ModelNet40Hdf5](../api/datasets/modelnet.md)<br><span class="tp-index__detail">:arxiv: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)</span> | ~12k | 40 |
| [ScanObjectNN](../api/datasets/scanobjectnn.md)<br><span class="tp-index__detail">:arxiv: [Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data](https://arxiv.org/abs/1908.04616)</span> | 2.9k | 15 |

</div>

### Indoor scene segmentation

<div class="tp-index" markdown>

| Dataset | Scenes | Classes |
| --- | ---: | ---: |
| [S3DIS](../api/datasets/s3dis.md)<br><span class="tp-index__detail">:cvf: [3D Semantic Parsing of Large-Scale Indoor Spaces](https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf)</span> | 272 rooms, 6 areas | 13 |
| [ScanNet v2](../api/datasets/scannet.md)<br><span class="tp-index__detail">:arxiv: [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](https://arxiv.org/abs/1702.04405)</span> | 1.5k scenes | 20 (NYU40) / 200 |

</div>

### Outdoor / driving segmentation

<div class="tp-index" markdown>

| Dataset | Frames | Classes |
| --- | ---: | ---: |
| [SemanticKITTI](../api/datasets/semantickitti.md)<br><span class="tp-index__detail">:arxiv: [SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences](https://arxiv.org/abs/1904.01416)</span> | 43k frames | 19 |
| [Semantic3D](../api/datasets/semantic3d.md)<br><span class="tp-index__detail">:arxiv: [Semantic3D.net: A new Large-scale Point Cloud Classification Benchmark](https://arxiv.org/abs/1704.03847)</span> | 30 scenes | 8 |
| [Toronto3D](../api/datasets/toronto3d.md)<br><span class="tp-index__detail">:arxiv: [Toronto-3D: A Large-scale Mobile LiDAR Dataset for Semantic Segmentation of Urban Roadways](https://arxiv.org/abs/2003.08284)</span> | 4 areas | 8 |
| [ParisLille3D](../api/datasets/parislille3d.md)<br><span class="tp-index__detail">:arxiv: [Paris-Lille-3D: a large and high-quality ground truth urban point cloud dataset for automatic segmentation and classification](https://arxiv.org/abs/1712.00032)</span> | 3 scenes | 9 |
| [DALES](../api/datasets/dales.md)<br><span class="tp-index__detail">:arxiv: [DALES: A Large-scale Aerial LiDAR Data Set for Semantic Segmentation](https://arxiv.org/abs/2004.11985)</span> | 40 tiles | 8 |
| [FRACTAL](../api/datasets/fractal.md)<br><span class="tp-index__detail">:arxiv: [FRACTAL: An Ultra-Large-Scale Aerial Lidar Dataset for 3D Semantic Segmentation of Diverse Landscapes](https://arxiv.org/abs/2405.04634)</span> | 100k patches | 7 |

</div>

### Part segmentation

<div class="tp-index" markdown>

| Dataset | Samples | Classes |
| --- | ---: | ---: |
| [ShapeNetPart](../api/datasets/shapenetpart.md)<br><span class="tp-index__detail">[A Scalable Active Framework for Region Annotation in 3D Shape Collections](https://dl.acm.org/doi/10.1145/2980179.2980238)</span> | ~16k | 16 categories / 50 parts |

</div>

### Registration

<div class="tp-index" markdown>

| Dataset | Pairs | Benchmarks |
| --- | ---: | ---: |
| [ThreeDMatch](../api/datasets/threedmatch.md)<br><span class="tp-index__detail">:arxiv: [3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions](https://arxiv.org/abs/1603.08182)</span> | 20.6k train / 1.6k + 1.8k test | 3DMatch, 3DLoMatch |

</div>

### Base class

<div class="tp-index" markdown>

| Dataset | Task |
| --- | ---: |
| [PointCloudDataset](../api/datasets/pointcloud.md)<br><span class="tp-index__detail">Base class of the loaders: the `root/<name>/raw` and `processed` paths and their existence checks. It has no download, process or transform hook; a loader implements `__getitem__` and `__len__`, and any `torch.utils.data.Dataset` returning sample dicts works as well.</span> | (any) |

</div>

## Dict keys

Every loader returns sample dicts keyed by `DataKeys` (`torch_pointcloud.utils.data`): `pos`, `color`, `normal`,
`segment`, `instance`, `label`, `box`, `face`, ... The [data format](../data-format.md) page lists each key with its
shape and dtype, and how `collate` batches it; the task pages below note what each dataset adds.
