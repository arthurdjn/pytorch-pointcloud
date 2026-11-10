# About

:pytorch-pointcloud-mini: `torch-pointcloud` is a :pytorch: [PyTorch](https://pytorch.org) library for deep learning on point clouds: models, pretrained weights, datasets, transforms, losses, metrics and inferers. The design was heavily inspired by :pytorch: [timm](https://github.com/huggingface/pytorch-image-models) and :pyg: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). Point clouds expect packed batches, the PyG layout, so a model trained on one benchmark runs in another codebase without changing the tensor layout.

_This project is independent and has no affiliation with the PyTorch team._

## Scope

`torch-pointcloud` does not replace research codebases such as :github: [Pointcept](https://github.com/Pointcept/Pointcept), :github: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and :github: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).
Instead this project focuses on providing a common interface for different models to ease benchmarking and reusability of common architectures without reinventing the wheel every time.

## Design

- **Packed batches.** Every point cloud is $(N, \cdot)$ with a `batch` index, nothing is padded. See the
  [data format](data-format.md) page.
- **One factory and a registry.** `create_model(name, task=..., pretrained=...)` builds every model. A name reads
  `<arch>[-<size>][.<dataset>[.<author>]]` and ships weights when it ends with an author tag. `model_info` and
  `list_models` read the registry without building anything.
- **Dict transforms.** A transform reads and writes the keys it is given and returns a new dict, like
  :monai: [MONAI](https://docs.monai.io/). The preprocessing of a checkpoint is a transform like any other.
- **Free functions.** `ops`, `losses`, `metrics` and `optim` are functions over tensors and dicts, with the argument
  names of `sklearn`, `torchmetrics` and `torch.optim`.
- **Inferers.** Running a model on a scene larger than one forward pass (tiling, windows, test-time augmentation) is
  a reusable object.
- **Optional kernels.** `spconv`, `flash-attn`, `mamba`, `ocnn`, `torchsparse` and `sptr` are imported lazily. A model
  that needs one raises an `ImportError` naming it, and the rest works without it.

## Citation

If you find this project useful, please cite it as follows:

```bibtex
@software{dujardin2026pytorchpointcloud,
  author = {Dujardin, Arthur},
  title = {PyTorch PointCloud},
  year = {2026},
  url = {https://github.com/arthurdjn/pytorch-pointcloud},
  doi = {10.5281/zenodo.22159632},
  license = {Apache-2.0}
}
```

## License

The code is Apache-2.0, see [`LICENSE`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/LICENSE).
Pretrained weights keep the license of their source, listed in
[`THIRD_PARTY_NOTICES.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/THIRD_PARTY_NOTICES.md).

## Contributing

Bug reports and feature requests go through the [issues](https://github.com/arthurdjn/pytorch-pointcloud/issues),
fixes through pull requests. [`CONTRIBUTING.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CONTRIBUTING.md)
has the development setup and the checks a pull request runs.
[`CODE_OF_CONDUCT.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CODE_OF_CONDUCT.md) applies to every
contribution.
