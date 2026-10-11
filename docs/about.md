# About

:pytorch-pointcloud-mini: `torch-pointcloud` is a :pytorch: [PyTorch](https://pytorch.org) library for deep learning on
point clouds: models, pretrained weights, datasets, transforms, losses, metrics and inferers. The design follows
:github: [timm](https://github.com/huggingface/pytorch-image-models) for images and :pyg: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
for graphs, with a `create_model` factory, a registry of pretrained weights and composable transforms. Point clouds
move as packed batches, the PyG layout, so a model trained on one benchmark runs in another codebase without changing
the tensor layout.

The project is independent and has no affiliation with the PyTorch team.

## Scope

Every piece is importable on its own and works in a plain PyTorch loop. The
[examples](https://github.com/arthurdjn/pytorch-pointcloud/tree/main/examples) reproduce the authors' recipes with
that API and nothing else. There is no config system in the library, and :lightning: [Lightning](https://lightning.ai/docs/pytorch/stable/)
stays an optional extra.

`torch-pointcloud` does not replace research codebases such as :github: [Pointcept](https://github.com/Pointcept/Pointcept),
:github: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and :github: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).
They own the configs and the experiment management. `torch-pointcloud` gives their architectures a common interface and
converts their checkpoints, so benchmarking across families is one script.

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

The concept DOI [10.5281/zenodo.22159632](https://doi.org/10.5281/zenodo.22159632) resolves to the latest release.
[`CITATION.cff`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CITATION.cff) has the same entry.

## License

The code is Apache-2.0, see [`LICENSE`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/LICENSE). Pretrained
weights keep the license of their source, stored in `model_info(name)["weights"]["license"]` and listed in
[`THIRD_PARTY_NOTICES.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/THIRD_PARTY_NOTICES.md). The
Sonata, Concerto and Utonia checkpoints are CC-BY-NC-4.0.

## Contributing

Bug reports and feature requests go through the [issues](https://github.com/arthurdjn/pytorch-pointcloud/issues),
fixes through pull requests. [`CONTRIBUTING.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CONTRIBUTING.md)
has the development setup and the checks a pull request runs.
[`CODE_OF_CONDUCT.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CODE_OF_CONDUCT.md) applies to every
contribution.
