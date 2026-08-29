# pytorch-pointcloud

<div align="center" style="width: 100%; margin: auto">
  <a href="https://pytorch-pointcloud.org/" rel="noopener"><img src="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/pytorch-pointcloud.png" alt="Banner"></a>

[![python](https://img.shields.io/badge/python-3.10+-red.svg?color=EE4C2C&labelColor=11001C&logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/pytorch-2.5+-red.svg?color=EE4C2C&labelColor=11001C&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![cuda](https://img.shields.io/badge/cuda-11.8+-red.svg?color=EE4C2C&labelColor=11001C&logo=nvidia&logoColor=white)](https://pytorch.org/)  
[![ruff](https://img.shields.io/badge/ruff-linter-red.svg?color=EE4C2C&labelColor=11001C&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff)
[![uv](https://img.shields.io/badge/uv-packaging-orange.svg?color=EE4C2C&labelColor=11001C&logo=uv&logoColor=white)](https://docs.astral.sh/uv)
[![mypy](https://img.shields.io/badge/mypy-typing-red.svg?color=EE4C2C&labelColor=11001C&logo=python&logoColor=white)](https://mypy-lang.org)
[![pytest](https://img.shields.io/badge/pytest-testing-red.svg?color=EE4C2C&labelColor=11001C&logo=pytest&logoColor=white)](https://pytest.org)

</div>

<p align="center">
A PyTorch library for deep learning on point clouds: models, pretrained weights, datasets, transforms and inferers,
behind one <code>create_model</code> factory in the spirit of <a href="https://github.com/huggingface/pytorch-image-models">timm</a>.
</p>

<br>
<br>

<table align="center">
  <tr>
    <td align="center" width="33%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/classification_dark.webp">
        <img src="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/classification.webp" alt="A chair turning a full circle, classified as a chair" width="100%">
      </picture><br>
      <b>Object classification</b><br><code>pointnet2-ssg.modelnet40.xu-yan</code>
    </td>
    <td align="center" width="33%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/part_segmentation_dark.webp">
        <img src="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/part_segmentation.webp" alt="An airplane turning a full circle, its parts colored by class" width="100%">
      </picture><br>
      <b>Part segmentation</b><br><code>pointnext-sm.shapenetpart.openpoints</code>
    </td>
    <td align="center" width="33%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/indoor_dark.webp">
        <img src="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/indoor.webp" alt="A camera gliding through a scanned house, every point colored by semantic class" width="100%">
      </picture><br>
      <b>Indoor segmentation / detection</b><br><code>ptv3-base.scannet20.pointcept</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/driving_dark.webp">
        <img src="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/driving.webp" alt="A bird's-eye camera riding down a LiDAR sequence with segmented points and detected boxes" width="100%">
      </picture><br>
      <b>Outdoor segmentation / detection</b><br><code>spvcnn-119gmacs.semantickitti.mit-han-lab</code><br><code>second.kitti.openpcdet</code>
    </td>
    <td align="center" width="33%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/survey_dark.webp">
        <img src="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/survey.webp" alt="A slow turn around the Eiffel Tower as an airborne LiDAR survey, colored by embedding" width="100%">
      </picture><br>
      <b>Large scale segmentation</b><br><code>utonia-lp.scannet20.pointcept</code>
    </td>
    <td align="center" width="33%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/similarity_dark.webp">
        <img src="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/animations/hero/similarity.webp" alt="The same house seen by a self-supervised encoder, points lit by similarity to a query" width="100%">
      </picture><br>
      <b>Features extraction</b><br><code>sonata-lp.scannet20.fair</code>
    </td>
  </tr>
</table>

<br>

## 🎉 Highlights

- **35 architectures** for classification, part and semantic segmentation, 3D detection and
  self-supervised pretraining: PointNet, PointNet++, DGCNN, KPConv, RandLA-Net, PointNeXt, PointMLP, PointConv,
  PVCNN, Point Transformer V1/V2/V3, SpUNet, SPVCNN, OctFormer, SphereFormer, Sonata, Concerto, Utonia, Point-MAE,
  Point-BERT, PointGPT, PointMamba, VoteNet, 3DETR, PointPillars, SECOND, PointRCNN, VoxelNeXt, LION and
  more.
- **134 pretrained checkpoints**, each carrying its benchmark metrics measured through this library with the
  reference protocol, and loaded with a single `create_model(..., pretrained=True)`.
- **Datasets** with download and preprocessing: ModelNet40, ScanObjectNN, ShapeNetPart, S3DIS, ScanNet, SemanticKITTI,
  nuScenes, KITTI, SUN RGB-D, Paris-Lille-3D, Semantic3D and Toronto3D.
- **Transforms** as MONAI-style dict transforms with tensor-level functional equivalents, and **inferers** for the
  usual evaluation protocols (test-time augmentation, voxel partition, sliding window, potential sphere voting).
- **Packed batches** everywhere: a flat $(N, \ldots)$ tensor plus a $(N,)$ batch index, never padded tensors.
- **Fully typed** (mypy strict), Python 3.10 to 3.13, an optional Lightning integration.

<br>

## 📦 Installation

```bash
pip install torch-pointcloud
```

The CUDA extensions (PyG kernels, spconv, flash-attention, Mamba, ocnn, torchsparse) are optional and only needed by
the architectures that use them.
See the [Installation](https://pytorch-pointcloud.org/installation/) page for the exact install command.

<br>

## 🚀 Quickstart

```python
import torch
import torch_pointcloud as tp

# Requires torch-cluster, torch-scatter
model = tp.create_model(
    "pointnext-sm.scanobjectnn.openpoints",
    task="classification",
    pretrained=True,
).eval()

pos = torch.randn(2048, 3)  # (N, 3) coordinates
x = torch.cat([pos, pos[:, 1:2] - pos[:, 1].min()], dim=1)  # (N, 4) features: xyz + height
batch = torch.zeros(2048, dtype=torch.long)  # (N,) batch index

with torch.no_grad():
    logits = model(x, pos, batch)  # (1, 15)
```

Every checkpoint ships the transform that turns a raw point cloud into what the network expects:

```python
# Requires torch-scatter, torch-cluster, spconv
model, info = tp.create_model("ptv3-base.scannet20.pointcept", task="segmentation", pretrained=True, return_info=True)
info["transform"]  # the preprocessing pipeline of that checkpoint
info["weights"]["metrics"]  # {"mIoU": 76.29}

tp.list_models("pointnext*")  # every registered PointNeXt config
tp.list_models(task="detection", pretrained=True)  # all detection checkpoints
```

The [examples](examples/) directory for hands-on usage (benchmarks and training recipes).

<br>

## 📚 Documentation

The [documentation](https://pytorch-pointcloud.org/) covers [installation](https://pytorch-pointcloud.org/installation/), a [get-started](https://pytorch-pointcloud.org/get-started/) guide,
the [model zoo](https://pytorch-pointcloud.org/models/overview/), [datasets](https://pytorch-pointcloud.org/datasets/overview/), [transforms](https://pytorch-pointcloud.org/transforms/overview/),
tutorials and the full API reference.

<br>

## 📝 Citation

If you find this project useful, please consider citing:

```bibtex
@software{dujardin2026pytorchpointcloud,
  author = {Dujardin, Arthur},
  title = {PyTorch PointCloud},
  year = {2026},
  url = {https://github.com/arthurdjn/pytorch-pointcloud},
  doi = {10.5281/zenodo.22159633},
  license = {Apache-2.0}
}
```

<br>

## 📄 License

Apache 2.0. See [LICENSE](LICENSE).
