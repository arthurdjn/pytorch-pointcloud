# pytorch-pointcloud

PyTorch models, datasets, transforms, and pretrained weights for 3D point cloud deep learning.
The design mirrors [timm](https://github.com/huggingface/pytorch-image-models): a `create_model` factory,
a pretrained-weight registry, and composable dict transforms, built on
[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)'s packed-batch format.

## Highlights

- **37 architectures** for classification, segmentation, detection, and self-supervised pretraining:
  PointNet, PointNet++, DGCNN, KPConv, RandLA-Net, PointNeXt, PointMLP, Point Transformer V1/V2/V3,
  SpUNet, SPVCNN, OctFormer, SphereFormer, OneFormer3D, Sonata, Point-MAE, VoteNet, 3DETR,
  PointPillars, SECOND, and more.
- **Pretrained weights** verified against the reference implementations.
- **Datasets** with download and preprocessing: ModelNet, ScanNet, S3DIS, ScanObjectNN, ShapeNetPart,
  SemanticKITTI, SunRGBD, Toronto3D, and more.
- **Dict transforms** (MONAI-style) with tensor-level functional equivalents.
- **Fully typed** (mypy strict), tested on Python 3.10 to 3.13.

## Installation

Not yet published to PyPI. Clone the repository, then either:

```bash
# Full CUDA environment (pinned torch + sparse-conv extensions)
# If using torchsparse, you need google-sparsehash:
# sudo apt-get install libsparsehash-dev
bash ./install.sh

# or a plain environment without the CUDA extras
uv sync
```

See the [installation guide](docs/installation.md) for the CUDA compatibility matrix and optional extras.

Dataset roots and cache locations are configured through `TORCH_POINTCLOUD_*` environment variables,
loaded from a `.env` file at the project root. Copy [.env.example](.env.example) and adjust the paths.

## Quickstart

```python
import torch
import torch_pointcloud as tp

model = tp.create_model("pointnext-sm.scanobjectnn.openpoints", task="classification", pretrained=True).eval()

pos = torch.randn(2048, 3)  # (N, 3) coordinates
x = torch.cat([pos, pos[:, 1:2] - pos[:, 1].min()], dim=1)  # (N, 4) features: xyz + height
batch = torch.zeros(2048, dtype=torch.long)  # (N,) batch index

with torch.no_grad():
    logits = model(x, pos, batch)  # (1, 15)
```

Models accept packed batches: a flat $(N, ...)$ tensor plus a $(N,)$ `batch` index, never padded
$(B, N, ...)$ tensors. Discover registered names (and which ones ship weights) with `list_models`:

```python
tp.list_models("pointnext*")                          # every registered PointNeXt config
tp.list_models(task="segmentation", pretrained=True)  # all segmentation checkpoints
```

See the [get-started guide](docs/get-started.md) for datasets, transforms, and training.

## Documentation

Build and serve the docs locally:

```bash
make docs   # strict build (regenerates the API reference)
make serve  # live-reload at 127.0.0.1:8000
```

## License

Apache 2.0. See [LICENSE](LICENSE).
