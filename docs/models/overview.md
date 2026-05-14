# Models

`torch-pointcloud` ships 22 architectures across point cloud classification, segmentation, and self-supervised pretraining. Every model is registered with a timm-style factory:

```python
import torch_pointcloud as tp

model = tp.create_model("pointnext-base.scanobjectnn", pretrained=True)
```

Browse available checkpoints with `list_models(task="classification")` / `list_models(task="segmentation")` from `torch_pointcloud.models._registry`.

## Choose by task

### Classification

Best for **shape classification** (ModelNet40, ScanObjectNN, ShapeNet objects). Inputs are single object scans, outputs are scene-level class predictions.

| Model | Paper | Notes | API |
| --- | --- | --- | --- |
| **PointNet** | :arxiv: [Qi et al. 2017](https://arxiv.org/abs/1612.00593) | The original. MLP per point + global max-pool. Strong baseline. | [`pointnet`](../api/models/pointnet.md) |
| **PointNet++** | :arxiv: [Qi et al. 2017](https://arxiv.org/abs/1706.02413) | Hierarchical Set Abstraction with FPS and ball-query grouping. | [`pointnet2`](../api/models/pointnet2.md) |
| **DGCNN** | :arxiv: [Wang et al. 2018](https://arxiv.org/abs/1801.07829) | EdgeConv over dynamic kNN graphs in feature space. | [`dgcnn`](../api/models/dgcnn.md) |
| **PointCNN** | :arxiv: [Li et al. 2018](https://arxiv.org/abs/1801.07791) | $\chi$-transform learns a permutation to canonicalise neighbourhoods. | [`pointcnn`](../api/models/pointcnn.md) |
| **PointConv** | :arxiv: [Wu et al. 2018](https://arxiv.org/abs/1811.07246) | Continuous convolution with density-reweighted kernels. | [`pointconv`](../api/models/pointconv.md) |
| **PointMLP** | :arxiv: [Ma et al. 2022](https://arxiv.org/abs/2202.07123) | Pure-MLP residual blocks with geometric affine. Strong / cheap. | [`pointmlp`](../api/models/pointmlp.md) |
| **PointNeXt** | :arxiv: [Qian et al. 2022](https://arxiv.org/abs/2206.04670) | PointNet++ revisited with modern training. SOTA on ModelNet40 / ScanObjectNN. | [`pointnext`](../api/models/pointnext.md) |
| **Point Transformer V1** | :arxiv: [Zhao et al. 2020](https://arxiv.org/abs/2012.09164) | Self-attention with subtraction-based scoring. | [`point_transformer`](../api/models/point_transformer.md) |
| **Point Transformer V2** | :arxiv: [Wu et al. 2022](https://arxiv.org/abs/2210.05666) | Grouped vector attention + partition-based pooling. | [`point_transformer_v2`](../api/models/point_transformer_v2.md) |
| **Point Transformer V3** | :arxiv: [Wu et al. 2023](https://arxiv.org/abs/2312.10035) | Serialized point patches with Hilbert / Z-order; very fast. | [`point_transformer_v3`](../api/models/point_transformer_v3.md) |
| **PVCNN** | :arxiv: [Liu et al. 2019](https://arxiv.org/abs/1907.03739) | Point + voxel hybrid; efficient sparse-conv stem. | [`pvcnn`](../api/models/pvcnn.md) |
| **Point-Mamba** | :arxiv: [Liang et al. 2024](https://arxiv.org/abs/2402.10739) | State-space model alternative to attention. | [`point_mamba`](../api/models/point_mamba.md) |
| **OctFormer** | :arxiv: [Wang 2023](https://arxiv.org/abs/2305.03045) | Octree-based attention; scales to dense scenes. | [`octformer`](../api/models/octformer.md) |

### Segmentation

Best for **dense per-point labelling** (S3DIS, ScanNet, SemanticKITTI). Inputs are large scenes, outputs are per-point class predictions.

| Model | Paper | Notes | API |
| --- | --- | --- | --- |
| **PointNet++** | :arxiv: [Qi et al. 2017](https://arxiv.org/abs/1706.02413) | Hierarchical SA / FP encoder-decoder. | [`pointnet2`](../api/models/pointnet2.md) |
| **DGCNN** | :arxiv: [Wang et al. 2018](https://arxiv.org/abs/1801.07829) | EdgeConv + U-Net style for part segmentation. | [`dgcnn`](../api/models/dgcnn.md) |
| **KPConv** | :arxiv: [Thomas et al. 2019](https://arxiv.org/abs/1904.08889) | Kernel Point Convolution with rigid / deformable kernels. | [`kpconv`](../api/models/kpconv.md) |
| **RandLA-Net** | :arxiv: [Hu et al. 2019](https://arxiv.org/abs/1911.11236) | Random sampling + local feature aggregation for outdoor scenes. | [`randlanet`](../api/models/randlanet.md) |
| **SPVCNN** | :arxiv: [Tang et al. 2020](https://arxiv.org/abs/2007.16100) | Sparse Point-Voxel Convolution; the reference fast indoor model. | [`spvcnn`](../api/models/spvcnn.md) |
| **SPUNet** | :arxiv: [Choy et al. 2019](https://arxiv.org/abs/1904.08755) | Pure 3D-UNet on sparse voxels; the Pointcept default backbone. | [`spunet`](../api/models/spunet.md) |
| **PVCNN++** | :arxiv: [Liu et al. 2019](https://arxiv.org/abs/1907.03739) | Segmentation variant of PVCNN. | [`pvcnn2`](../api/models/pvcnn2.md) |

### Self-supervised pretraining

| Model | Paper | Notes | API |
| --- | --- | --- | --- |
| **Sonata** | :github: [facebookresearch/sonata](https://github.com/facebookresearch/sonata) | Self-supervised pretraining on diverse 3D corpora. | [`sontata`](../api/models/sontata.md) |
| **Concerto** | :arxiv: [Long et al. 2024](https://arxiv.org/abs/2510.23607) | Joint 2D-3D self-supervised learning. | [`concerto`](../api/models/concerto.md) |
| **Utonia** | :arxiv: [Yang et al. 2024](https://arxiv.org/abs/2603.03283) | One encoder, many point clouds. | [`utonia`](../api/models/utonia.md) |

## Choose by registered checkpoint name

The convention is `<arch>-<size>.<dataset>` (e.g. `pointnext-base.scanobjectnn`). Use `list_models(task=...)` to enumerate everything currently registered:

```python
from torch_pointcloud.models._registry import list_models

for name in list_models(task="classification"):
    print(name)
```

## Picking a model

- **Small, simple, fast** (classification): start with `PointNet` or `PointMLP-elite`.
- **Modern accuracy, classification**: `PointNeXt-base` or `Point Transformer V3`.
- **Indoor scene segmentation**: `KPConv`, `SPVCNN`, or `Point Transformer V3`.
- **Outdoor / driving (large clouds)**: `RandLA-Net` or `SPVCNN`.
- **Sparse, dense scenes (rooms)**: `OctFormer` or `SPUNet`.

For each architecture, the API page links to the original paper and to the model factory.
