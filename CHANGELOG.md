# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.0.6 (2026-09-23)

- Renamed the ScanObjectNN checkpoints to name their split (`-hardest`, `-objbg`, `-objonly`) and the KPConv checkpoints to `s3dis-area5`.
- Pinned every registered weight URL to its Hub commit.
- Fixed the `point-mamba-base.scanobjectnn-objonly.dingkang-liang` weights (block norms taken from the unused `norm_ffn` layer): OA 83.30 to 92.60.
- Added `accept_terms` to `S3DIS`, `S3DISHdf5`, `ScanNet` and `ScanObjectNN`; `download` asks for it on the terminal otherwise.
- Updated `create_model(pretrained=True)` to raise when the model registers no weights.
- Added `torch_pointcloud.metrics` to the package namespace.
- Added the upstream copyright notices to `THIRD_PARTY_NOTICES.md`.
- Updated the documentation: install order, canonical `latest` links, social card, model overview scores.

## 0.0.5 (2026-09-21)

- Added performances of object detection models on nuScenes.
- Updated the minimum `torch` version to 2.8.
- Updated the CI to test on `torch` 2.8 and to run on pull requests.
- Added `torch` 2.8 to the install selector.
- Updated DGCNN, T-Net, PointNeXt and Point Transformer to use the library's `knn`, `knn_graph` and `radius`.
- Added `CONTRIBUTING.md` and issue templates.
- Added the dependencies and their licenses to `THIRD_PARTY_NOTICES.md`.
- Updated the README with a warning on the required PyG kernels.
- Added social cards to the documentation.
- Added `examples/benchmark.sh` to run every benchmark.
- Removed the `pointnext-xl.s3dis-area6.openpoints` registration (no weights).
- Removed the out-of-memory skip of the PTv3 benchmark.
- Updated the benchmark results after the inferers seed fix.
- Added the secondary metrics (`mAcc`, `OA`, per-class AP) to the registry.

## 0.0.4 (2026-09-20)

- Added versioning of the documentation with mike.
- Moved the metrics from `torch_pointcloud.utils.metrics` to the `torch_pointcloud.metrics` sub-package.
- Added `intersection_over_union` and `accuracy` of a `confusion_matrix`, with `average`, `ignore_index`, `class_names`.
- Removed `compute_iou`, `compute_mean_iou`, `overall_accuracy` and `per_class_accuracy`, superseded by the above.
- Renamed `part_iou` to `part_intersection_over_union`; `part_mean_intersection_over_union` scores it with `average`.
- Added `box_matches`; `average_precision3d` now scores its records, with `average` and `class_names`.
- Removed `mean_average_precision3d`, superseded by `average_precision3d`.
- Updated `instance_average_precision` to take `iou_threshold`, `average` and `class_names`, like the other metrics.
- Updated box AP and `count_points_in_boxes` for faster detection benchmarks.
- Added `inverse_key` to `VoxelPartitionInferer`, `KNNWindowInferer` and `PotentialSphereInferer`, so a `transform`
  that changes the number of points (pad, voxelize) works in every inferer, as it did in `SlidingWindowInferer`.
- Renamed `VoxelPartitionInferer(reduce=...)` to `aggregate`, and the `"weighted_mean"` mode of `KNNWindowInferer`
  to `"mean"`, so every inferer names its aggregation the same way.
- Renamed `TTAInferer(ema_softmax=...)` to `softmax`, now `False` by default and applied in both aggregation modes.
- Added `TTAInferer(transforms=None, num_passes=...)` to repeat the base inferer without augmentation.
- Added `DataKeys.BLOCK_BBOX`, the key `SlidingWindowInferer` writes the block bounds under.
- Fixed seeded inferers drawing the same fragments on every call, which made the votes of a `TTAInferer` identical:
  an int `seed` now advances on each call, and `seed=None` follows the global generator like the transforms.
- Updated the `pointnet2.s3dis-area5.xu-yan` benchmark to 54.83 mIoU (from 54.28), now that its three votes differ.
- Updated the inferers overview in the docs.
- Renamed `serialize_coords` to `serialize_pos` and `RelativePositionalEncoding.coords_boundary` to `pos_boundary`.

## 0.0.3 (2026-09-19)

- Fixed SPUNet and PTv3 color normalizations to reproduce the same processing as original.
- Updated all tests datasets to use random data instead of subsampled versions of original ones (for testing).
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
