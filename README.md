# pytorch-pointcloud

<div align="center" style="width: 100%; margin: auto">
  <a href="https://pytorch-pointcloud.org/" rel="noopener"><img src="https://raw.githubusercontent.com/arthurdjn/pytorch-pointcloud/main/docs/assets/pytorch-pointcloud.png" alt="Banner"></a>

[![python](https://img.shields.io/pypi/pyversions/torch-pointcloud?color=EE4C2C&labelColor=11001C&logo=python&logoColor=white)](https://pypi.org/project/torch-pointcloud/)
[![pytorch](https://img.shields.io/badge/pytorch-2.5+-red.svg?color=EE4C2C&labelColor=11001C&logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![tests](https://img.shields.io/github/actions/workflow/status/arthurdjn/pytorch-pointcloud/test.yml?branch=main&label=tests&labelColor=11001C&logo=github&logoColor=white)](https://github.com/arthurdjn/pytorch-pointcloud/actions/workflows/test.yml)
[![pypi](https://img.shields.io/pypi/v/torch-pointcloud?color=EE4C2C&labelColor=11001C&logo=pypi&logoColor=white)](https://pypi.org/project/torch-pointcloud/)
[![license](https://img.shields.io/badge/license-Apache--2.0-red.svg?color=EE4C2C&labelColor=11001C)](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/LICENSE)
[![doi](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22159633-red.svg?color=EE4C2C&labelColor=11001C)](https://doi.org/10.5281/zenodo.22159633)
[![docs](https://img.shields.io/badge/docs-pytorch--pointcloud.org-red.svg?color=EE4C2C&labelColor=11001C&logo=materialformkdocs&logoColor=white)](https://pytorch-pointcloud.org/)  

</div>

<p align="center">
A PyTorch library for deep learning on point clouds: models, pretrained weights, datasets, transforms and inferers,
behind one <code>create_model</code> factory, inspired by <a href="https://github.com/huggingface/pytorch-image-models">timm</a>.
<br>
<br>
<i>Check out the official docs at <a href="https://www.pytorch-pointcloud.org">pytorch-pointcloud.org</a>!</i>
</p>

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

## Installation

Install the library with `pip` (or `uv`):

```bash
pip install torch-pointcloud
```

> [!IMPORTANT]
> The CUDA extensions (PyG kernels, spconv, flash-attention, Mamba, ocnn, torchsparse) are optional and only needed by
> the architectures that use them.
> See the [Installation](https://pytorch-pointcloud.org/installation/) page for the exact install command.

<br>

## Quickstart

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

## Documentation

The [documentation](https://pytorch-pointcloud.org/) covers [installation](https://pytorch-pointcloud.org/installation/), a [get-started](https://pytorch-pointcloud.org/get-started/) guide,
the [model zoo](https://pytorch-pointcloud.org/models/overview/), [datasets](https://pytorch-pointcloud.org/datasets/overview/), [transforms](https://pytorch-pointcloud.org/transforms/overview/),
tutorials and the full API reference.

<br>

## Citation

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

## License

Apache 2.0. See [LICENSE](LICENSE).
