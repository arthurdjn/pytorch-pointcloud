# About

:pytorch-pointcloud-mini: `torch-pointcloud` is to point clouds what :github: [timm](https://github.com/huggingface/pytorch-image-models) is to images: one
`create_model` call builds any registered architecture, loads its pretrained weights, and hands back the transform that
turns a raw cloud into what the network expects. The data moves in the packed batches of :pyg: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/),
so a model trained on one benchmark plugs into another codebase without a change of tensor layout.

The library is built for :pytorch: [PyTorch](https://pytorch.org), not by it: it is an independent project with no
affiliation to the PyTorch team.

## What it is, and is not

It is a library: models, pretrained weights, datasets, transforms, losses, metrics and inferers, each importable on its
own and usable in a plain PyTorch loop. The [examples](https://github.com/arthurdjn/pytorch-pointcloud/tree/main/examples)
reproduce the authors' recipes with that API and nothing else.

It is not a training framework or a research codebase. :github: [Pointcept](https://github.com/Pointcept/Pointcept),
:github: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and :github: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
own the configuration systems and the experiment management; `torch-pointcloud` gives their architectures a common interface
and converts their checkpoints, so that benchmarking across families is one script.

## Design principles

- **Packed batches.** Every point cloud is $(N, \cdot)$ with a `batch` index; nothing is padded. The
  [data format](data-format.md) page is the contract.
- **One factory and a registry.** `create_model(name, task=..., pretrained=...)` builds every model; a registered
  name reads `<arch>[-<size>][.<dataset>[.<author>]]` and ships weights exactly when it ends in an author tag.
  `model_info` and `list_models` read the registry without building anything.
- **Dict transforms.** A transform reads and writes the keys it is given and returns a new dict, in the style of
  :monai: [MONAI](https://docs.monai.io/); the checkpoint's preprocessing is a transform like any other.
- **Free functions.** `ops`, `losses`, `metrics` and `optim` are functions over plain tensors and dicts, with the
  argument names of `sklearn`, `torchmetrics` and `torch.optim`, so nothing has to be learned to call them.
- **Inferers.** How a model runs on a scene larger than one forward pass (tiling, windows, test-time augmentation,
  refinement) is a reusable object, not a loop inside every script.
- **Optional kernels.** `spconv`, `flash-attn`, `mamba`, `ocnn`, `torchsparse` and `sptr` are imported lazily; a
  model that needs one raises an `ImportError` naming it, and everything else keeps working without it.

## Roadmap

- One encoder and head contract across every family, so any backbone pairs with any task head.
- More checkpoints and datasets, each with its reference protocol reproduced and measured.
- Faster neighbor search, sampling and voxelization, on GPU and CPU.
- A training recipe in `examples/` for every checkpoint in the zoo.
- Experiment configurations and Lightning modules for every model, kept optional.

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

The concept DOI [10.5281/zenodo.22159632](https://doi.org/10.5281/zenodo.22159632) resolves to the latest release;
[`CITATION.cff`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CITATION.cff) carries the same entry.

## License

The code is Apache-2.0, see [`LICENSE`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/LICENSE). Pretrained
weights keep the license of their source, recorded in `model_info(name)["weights"]["license"]` and listed in
[`THIRD_PARTY_NOTICES.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/THIRD_PARTY_NOTICES.md); the Sonata,
Concerto and Utonia checkpoints are CC-BY-NC-4.0.

## Contributing

Bug reports and features go through [issues](https://github.com/arthurdjn/pytorch-pointcloud/issues) and pull
requests. [`CONTRIBUTING.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CONTRIBUTING.md) has the
development setup and the checks a pull request runs;
[`CODE_OF_CONDUCT.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CODE_OF_CONDUCT.md) applies to every
contribution.
