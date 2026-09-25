# Models

:pytorch-pointcloud-mini: `torch-pointcloud` ships architectures for point cloud classification, semantic and part
segmentation, object detection and self-supervised pretraining, behind one :pytorch: timm-style factory:

```{.python notest}
import torch_pointcloud as tp

model = tp.create_model("pointnext-sm.scanobjectnn-hardest.openpoints", pretrained=True)
tp.list_models(task="classification", pretrained=True)  # or "semantic-segmentation", "detection", ...
```

Weights are downloaded from the Hugging Face Hub on first use and cached under `~/.cache/torch-pointcloud/models`
(`TORCH_POINTCLOUD_MODELS_DIR` moves them). Each checkpoint keeps the license of its source, listed on the task pages,
and a few are restricted to non-commercial use: see
[`THIRD_PARTY_NOTICES.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/THIRD_PARTY_NOTICES.md).

![Five pretrained checkpoints on two committed sample clouds: object classification, part segmentation, scene segmentation, 3D detection, and LiDAR segmentation](../assets/tasks/hero.png)

## Tasks

<div class="grid cards" markdown>

-   :material-shape-outline: __[Classification](classification.md)__

    One label per cloud: run, evaluate and fine-tune a classifier.

-   :material-floor-plan: __[Semantic segmentation](segmentation.md)__

    One label per point: voxelization, full-resolution predictions, mIoU.

-   :material-puzzle-outline: __[Part segmentation](part-segmentation.md)__

    Category-conditioned part labels and the ShapeNetPart protocol.

-   :material-cube-scan: __[Object detection](detection.md)__

    Oriented boxes: decode, filter with NMS, score with mAP.

-   :material-palette-swatch: __[Feature maps](features.md)__

    The representation under the head: embeddings, retrieval, PCA.

</div>

### Classification

Best for **shape classification** (ModelNet40, ScanObjectNN, ShapeNet objects). Inputs are single object scans, outputs are scene-level class predictions.

| Model                                                             | Paper                                                                                                                                    |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [**PointNet**](../api/models/pointnet.md)                         | :arxiv: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)                 |
| [**PointNet++**](../api/models/pointnet2.md)                      | :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)               |
| [**DGCNN**](../api/models/dgcnn.md)                               | :arxiv: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)                                               |
| [**PointCNN**](../api/models/pointcnn.md)                         | :arxiv: [PointCNN: Convolution On $\mathcal{X}$-Transformed Points](https://arxiv.org/abs/1801.07791)                                    |
| [**PointConv**](../api/models/pointconv.md)                       | :arxiv: [PointConv: Deep Convolutional Networks on 3D Point Clouds](https://arxiv.org/abs/1811.07246)                                    |
| [**PointMLP**](../api/models/pointmlp.md)                         | :arxiv: [Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework](https://arxiv.org/abs/2202.07123) |
| [**PointNeXt**](../api/models/pointnext.md)                       | :arxiv: [PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies](https://arxiv.org/abs/2206.04670)               |
| [**Point Transformer V1**](../api/models/point_transformer.md)    | :arxiv: [Point Transformer](https://arxiv.org/abs/2012.09164)                                                                            |
| [**Point Transformer V2**](../api/models/point_transformer_v2.md) | :arxiv: [Point Transformer V2: Grouped Vector Attention and Partition-based Pooling](https://arxiv.org/abs/2210.05666)                   |
| [**Point Transformer V3**](../api/models/point_transformer_v3.md) | :arxiv: [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)                                              |
| [**PVCNN**](../api/models/pvcnn.md)                               | :arxiv: [Point-Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/abs/1907.03739)                                               |
| [**PointMamba**](../api/models/point_mamba.md)                    | :arxiv: [PointMamba: A Simple State Space Model for Point Cloud Analysis](https://arxiv.org/abs/2402.10739)                              |
| [**OctFormer**](../api/models/octformer.md)                       | :arxiv: [OctFormer: Octree-based Transformers for 3D Point Clouds](https://arxiv.org/abs/2305.03045)                                     |

### Segmentation

Best for **dense per-point labeling** (S3DIS, ScanNet, SemanticKITTI). Inputs are large scenes, outputs are per-point class predictions.

| Model                                                             | Paper                                                                                                                      |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [**PointNet++**](../api/models/pointnet2.md)                      | :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) |
| [**DGCNN**](../api/models/dgcnn.md)                               | :arxiv: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)                                 |
| [**KPConv**](../api/models/kpconv.md)                             | :arxiv: [KPConv: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/abs/1904.08889)                   |
| [**PointNeXt**](../api/models/pointnext.md)                       | :arxiv: [PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies](https://arxiv.org/abs/2206.04670) |
| [**Point Transformer V3**](../api/models/point_transformer_v3.md) | :arxiv: [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)                                |
| [**RandLA-Net**](../api/models/randlanet.md)                      | :arxiv: [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/abs/1911.11236)        |
| [**SPVCNN**](../api/models/spvcnn.md)                             | :arxiv: [Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution](https://arxiv.org/abs/2007.16100)       |
| [**SPUNet**](../api/models/spunet.md)                             | :arxiv: [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://arxiv.org/abs/1904.08755)           |
| [**OctFormer**](../api/models/octformer.md)                       | :arxiv: [OctFormer: Octree-based Transformers for 3D Point Clouds](https://arxiv.org/abs/2305.03045)                       |
| [**SphereFormer**](../api/models/sphereformer.md)                 | :arxiv: [Spherical Transformer for LiDAR-based 3D Recognition](https://arxiv.org/abs/2303.12766)                           |
| [**PVCNN / PVCNN++**](../api/models/pvcnn2.md)                    | :arxiv: [Point-Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/abs/1907.03739)                                 |
| [**SPFormer-UNet**](../api/models/spformer_unet.md)               | :arxiv: [Superpoint Transformer for 3D Scene Instance Segmentation](https://arxiv.org/abs/2211.15766)                      |

### Detection

Predict **3D bounding boxes** for indoor scenes (ScanNet, SUN RGB-D) or driving scenes (KITTI, nuScenes, Waymo).

| Model                                             | Paper                                                                                                                            |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [**VoteNet**](../api/models/votenet.md)           | :arxiv: [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/abs/1904.09664)                            |
| [**3DETR**](../api/models/threedetr.md)           | :arxiv: [An End-to-End Transformer Model for 3D Object Detection](https://arxiv.org/abs/2109.08141)                              |
| [**PointPillars**](../api/models/pointpillars.md) | :arxiv: [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)                   |
| [**SECOND**](../api/models/second.md)             | :arxiv: [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337)                           |
| [**PointRCNN**](../api/models/pointrcnn.md)       | :arxiv: [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/abs/1812.04244)              |
| [**VoxelNeXt**](../api/models/voxelnext.md)       | :arxiv: [VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking](https://arxiv.org/abs/2303.11301)                |
| [**Voxel-Mamba**](../api/models/voxel_mamba.md)   | :arxiv: [Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection](https://arxiv.org/abs/2406.10700) |
| [**LION**](../api/models/lion.md)                 | :arxiv: [LION: Linear Group RNN for 3D Object Detection in Point Clouds](https://arxiv.org/abs/2407.18232)                       |

### Self-supervised pretraining

Backbones pretrained without labels, registered with `task="pretraining"`; the fine-tuned classification / segmentation heads are registered under their downstream task.

| Model                                          | Paper                                                                                                                             |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| [**Point-MAE**](../api/models/point_mae.md)    | :arxiv: [Masked Autoencoders for Point Cloud Self-supervised Learning](https://arxiv.org/abs/2203.06604)                          |
| [**Point-BERT**](../api/models/point_bert.md)  | :arxiv: [Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling](https://arxiv.org/abs/2111.14819)       |
| [**Point-M2AE**](../api/models/point_m2ae.md)  | :arxiv: [Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training](https://arxiv.org/abs/2205.14401) |
| [**PointGPT**](../api/models/pointgpt.md)      | :arxiv: [PointGPT: Auto-regressively Generative Pre-training from Point Clouds](https://arxiv.org/abs/2305.11487)                 |
| [**PointMamba**](../api/models/point_mamba.md) | :arxiv: [PointMamba: A Simple State Space Model for Point Cloud Analysis](https://arxiv.org/abs/2402.10739)                       |
| [**Sonata**](../api/models/sonata.md)          | :arxiv: [Sonata: Self-Supervised Learning of Reliable Point Representations](https://arxiv.org/abs/2503.16429)                    |
| [**Concerto**](../api/models/concerto.md)      | :arxiv: [Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](https://arxiv.org/abs/2510.23607)        |
| [**Utonia**](../api/models/utonia.md)          | :arxiv: [Utonia: Toward One Encoder for All Point Clouds](https://arxiv.org/abs/2603.03283)                                       |
