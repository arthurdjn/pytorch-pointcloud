# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Configured heads from `configure_head` are now always in the same mode the current model is in (e.g. `train` or `eval`).
- Fixed `decimate_indices` such that when a generator is provided, it is reseeded for each sample.
- Updated tutorials docs summaries to better highlight the cards.
- Updated examples scripts to auto adjust number of classes based on specified dataset.
- Added tests in CI for all Python versions supported by the project (3.10-3.13).

## 0.0.1 (2026-08-29)

Initial release: a timm-style `create_model` factory and pretrained-weight registry covering
classification, segmentation, detection, self-supervised, and generative point cloud models,
built on packed PyG-style batches. Ships datasets with disk caching, MONAI-style dict transforms,
tiling and sliding-window inferers, optional PyTorch Lightning modules, and one training and one benchmark
script per model under `examples/`.
