# pytorch-pointcloud

<div align="center" style="width: 100%; margin: auto">
  <a href="" rel="noopener"><img src="docs/assets/pytorch-pointcloud.png" alt="Banner"></a>

[![python](https://img.shields.io/badge/python-3.10+-red.svg?color=EE4C2C&labelColor=11001C&logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/pytorch-2.5+-red.svg?color=EE4C2C&labelColor=11001C&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![cuda](https://img.shields.io/badge/cuda-11.8+-red.svg?color=EE4C2C&labelColor=11001C&logo=nvidia&logoColor=white)](https://pytorch.org/)  
[![ruff](https://img.shields.io/badge/ruff-linter-red.svg?color=EE4C2C&labelColor=11001C&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff)
[![uv](https://img.shields.io/badge/uv-packaging-orange.svg?color=EE4C2C&labelColor=11001C&logo=uv&logoColor=white)](https://docs.astral.sh/uv)
[![mypy](https://img.shields.io/badge/mypy-typing-red.svg?color=EE4C2C&labelColor=11001C&logo=python&logoColor=white)](https://mypy-lang.org)
[![pytest](https://img.shields.io/badge/pytest-testing-red.svg?color=EE4C2C&labelColor=11001C&logo=pytest&logoColor=white)](https://pytest.org)

</div>

<p align="center">
    PyTorch Point Cloud models, scripts, pretrained weights -- PointNet, PointNet++, DGCNN, KPConv, RandLA-Net, SPConv, VoteNet, PointGroup, SPVCNN, 3DETR, PointTransformer and more
</p>

<br>

## 🎉 Highlights

- **36 architectures** for classification, segmentation, detection, and self-supervised pretraining:
  PointNet, PointNet++, DGCNN, KPConv, RandLA-Net, PointNeXt, PointMLP, Point Transformer V1/V2/V3, SpUNet, SPVCNN, OctFormer, SphereFormer, Sonata, Point-MAE, VoteNet, 3DETR, PointPillars, SECOND, and more.
- **Pretrained weights** verified against the reference implementations.
- **Datasets** with download and preprocessing: ModelNet, ScanNet, S3DIS, ScanObjectNN, ShapeNetPart, SemanticKITTI, SunRGBD, Toronto3D, and more.
- **Dict transforms** (MONAI-style) with tensor-level functional equivalents.
- **Fully typed** (mypy strict), supports Python 3.10 to 3.13.

<br>

## 📦 Installation

```bash
pip install torch-pointcloud
```

See the [Installation](docs/installation.md) guide for more details on how to install the optional CUDA extensions.

<br>

## 🚀 Quickstart

```python
import torch
import torch_pointcloud as tp

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

Models accept packed batches: a flat $(N, ...)$ tensor plus a $(N,)$ `batch` index, never padded
$(B, N, ...)$ tensors. Discover registered names (and which ones ship weights) with `list_models`:

```python
import torch_pointcloud as tp

tp.list_models("pointnext*")                          # every registered PointNeXt config
tp.list_models(task="segmentation", pretrained=True)  # all segmentation checkpoints
```

See the [Get Started](docs/get-started.md) guide for datasets, transforms, and training.

<br>

## 📝 Citation

If you find this project useful, please consider citing:

```bibtex
@article{pytorch-pointcloud,
  title={PyTorch PointCloud},
  author={Arthur Dujardin},
  journal={GitHub},
  year={2026}
}
```

<br>

## 📄 License

Apache 2.0. See [LICENSE](LICENSE).
