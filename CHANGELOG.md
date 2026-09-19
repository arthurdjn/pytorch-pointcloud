# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Added `sw_batch_size` parameter for `SlidingWindowInferer` for faster inference.
- Updated benchmark scripts to use `batch_size > 1` for faster benchmarking.
- Updated the `SerializedRPE` attention module operation to use less memory.
- Added `FeaturesDict` so that `return_intermediates=True` returns the same list for every model: the encoder stages
  from fine to coarse, each with `x`, `batch` and `pos` or `pos_grid`.
- Renamed the intermediates keys `features` to `x` and `inverse` to `pooling_inverse`.
- Updated SpUNet, SPFormer-UNet and SPVCNN `forward_features` to return the skips only with `return_intermediates=True`.
- Fixed `octformer-lg` stem channels, the model could not run a forward pass.
- Fixed `reset_classifier` ignoring `global_pool` for PVCNN and PVCNN2.
- Fixed Point-MAE and Point-M2AE segmentation `forward_head` returning dense logits but packed `pre_logits` features.
- Added source links to classes and functions in the docs.

## 0.0.2 (2026-09-16)

- Pinned torch version to 2.10 in the CI to match local development.
- Updated pypi package URLs metadata, README documentation with better CI badges.
- Removed unlinked tutorials from the docs for future releases.
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
