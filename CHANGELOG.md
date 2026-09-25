# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Added a checkpoint table to every model page of the docs, generated from the registry, and a comparison with the research codebases to the README.
- Removed `SoftmaxPool` and `LogSoftmaxPool`, and the `"softmax"` / `"log_softmax"` names of `create_pool`: both always raised.
- Keyed the weights cache by the pinned revision (`<checkpoint>/<revision>/<file>`), so a checkpoint pinned to new weights downloads them instead of loading the stale file. Weights cached by 0.0.7 and earlier download once more.
- Replaced the `torch-cluster`, `torch-scatter` and `torch-sparse` imports with their `torch_geometric` equivalents, and required `torch-geometric>=2.8`, so `pyg-lib` is the only PyG kernel package to install.

## 0.0.7 (2026-09-24)

- Renamed `CenterLoss` and `SparseCenterLoss` to `CenterPointLoss` and `SparseCenterPointLoss`, in `losses.centerpoint`.
- Renamed the loss weights to `<term>_weight` (the 3DETR `loss_*_weight`, TransFusion `hm_weight` / `bbox_weight` to `heatmap_weight` / `loc_weight`, VoteNet `loss_scale` to `loss_weight`) and the TransFusion `hungarian_*_cost` to `matcher_*_cost`, and made every loss argument after `num_classes` keyword-only.
- Renamed `average_precision3d` to `box_average_precision`, and the Lightning metrics `AveragePrecision3D`, `MeanAveragePrecision3D` and `InstancePartMeanIoU` to `BoxAveragePrecision`, `BoxMeanAveragePrecision` and `InstancePartMeanIntersectionOverUnion`.
- Updated `nuscenes_detection_metrics` to take `(preds, target)` like `box_matches`, and `confusion_matrix(ignore_index)` to accept several indices.
- Added `kitti_average_precision`, the KITTI 3D detection protocol used by the PointPillars and SECOND benchmarks.
- Renamed the `3detr_*` and `ptv3_*` examples after their model modules (`threedetr_*`, `point_transformer_v3_*`).
- Added the class names to the OctFormer and PTv3 ScanNet checkpoints, and `SCANNET200_CLASSES`.
- Made the dataset arguments after `root` keyword-only everywhere (ModelNet, ScanNet, ScanObjectNN and `NuScenes(split)` were positional).
- Updated `ParisLille3D`, `Toronto3D` and `ShapeNetPart` to default to the train split like the other datasets, and typed every `split` as a `Literal`.
- Renamed `RepeatDataset(loop)` to `k`.
- Updated `SlidingWindowInferer(softmax)` to default to `False` like the other inferers.
- Renamed `VoxelPartitionInferer(sub_batch_size)` to `sw_batch_size`, and added `progress` to `VoxelPartitionInferer` and `TTAInferer`.
- Made `import torch_pointcloud` lazy: subpackages load on first access, so importing `transforms` or `datasets` no longer loads the models and spconv.
- Renamed `BaseBEVBackbone`, `BaseBEVResBackbone`, `BasicBlock2d`, `AnchorHeadSingle`, `AnchorHeadMulti`, `MultiGroupSingleHead`, `PFNLayer` and `RPE` to `BEVBackbone`, `BEVResidualBackbone`, `ResidualBlock2d`, `AnchorHead`, `MultiGroupAnchorHead`, `AnchorGroupHead`, `PillarFeatureLayer` and `OctreeRelativePositionEncoding`.
- Moved `utils.ops.voxel_grid_fnv` to `utils.voxelization`, `utils.ops.knn_interpolate` to `utils.cluster`, and `layers.ensure_msg_list` / `ensure_msg_list_size` to `utils.conversion`, and renamed `utils.neighbors` to `utils.density`.
- Added `input_keys` to the registry, the batch keys each model's `forward` takes, read by `LitModel` by default, and `utils.data.select_inputs`.
- Made `task` optional in `create_model` when the name is registered under a single task.
- Added `ModelInfoDict`, the type of `create_model(..., return_info=True)` info, with the resolved `task` and `input_keys`, and typed the registered `transform` as a `Transform`.
- Moved the `return_intermediates` argument of `PointTransformerV3Encoder.forward` after `pos` and `condition`.
- Replaced `RandomDropout(p_drop)` with `drop_ratio_range`, the range the dropped fraction is drawn from on each call, and renamed `functional.random_dropout_mask(p_drop)` to `drop_ratio`.
- Renamed `Voxelize(method="pyg")` to `method="grid"` and `Voxelize(random_sample)` to `random_first`.
- Moved `EncodeVoteNetTargets` and `GenerateVoteLabels` to `torch_pointcloud.models.votenet`, and `angle_to_class`, `class_to_angle` and `class_to_size` to `torch_pointcloud.utils.box3d`.
- Moved `functional.split_batch` to `torch_pointcloud.layers.serialized_attention`, and renamed `functional.random_color_drop` to `color_drop`.
- Removed `Abs(inplace)`, since transforms never mutate their input.
- Removed the type aliases (`ReduceOp`, `RescaleMethod`, `ShiftMethod`, `VoxelMethod`, `VoxelReduce`, `VoxelPosReduce`) from the `torch_pointcloud.transforms` exports.
- Renamed `SubtractKey` and `DivideKey` to `SubtractItems` and `DivideItems`, and `RandomShift` to `RandomTranslate` (`shift_range` to `translation_range`, `functional.shift_boxes` to `translate_boxes`).
- Removed `AlignAxis`, the same as `Shift(method="min", axes=[k])`.
- Renamed the output-key arguments to `dst_*`: `names` of `CopyItems` / `RenameItems` and `normal_key` of `EstimateNormals` to `dst_keys`, and the output keys of `RandomSampleFaceVertices`, `BuildOctree`, `HardVoxelize`, `GenerateVoteLabels`, `EncodeVoteNetTargets` and `RelabelBoxes`.
- Renamed `HardVoxelize(feat_key)` to `feature_key`, like `BuildOctree`.
- Moved every function of `torch_pointcloud.transforms.functional` next to its transform in the themed modules; `functional` imports them all.
- Renamed `functional.abs` to `absolute`.
- Split `torch_pointcloud.transforms.transforms` into `base`, `sampling`, `masking`, `geometry`, `scaling`, `utils`, `voxelization`, `octree`, `augmentation`, `box` and `mixing`; `torch_pointcloud.transforms` exports the same names.
- Removed `DataKeys.SEMANTIC`, `REFLECTANCE` and `ROOM_MAX`: `ParisLille3D` emits `intensity` and the S3DIS blocks `scene_max`, like the other datasets.
- Renamed `DataKeys.POINTS` to `OCTREE_POINTS`.
- Renamed `num_group`, `num_heading_bin` (and `num_angle_bin`), `num_size_cluster` and `num_proposal` to their plurals in the models, losses, box transforms and `utils.cluster.group`.
- Renamed `sa_npoints`, `roi_sa_npoints` and `preenc_npoints` to `sa_num_points`, `roi_sa_num_points` and `preencoder_num_points`.
- Renamed the PointNet `mlp1_dims` / `mlp2_dims` to `mlp1_channels` / `mlp2_channels`, and the Point Transformer `encoder_num_groups` / `decoder_num_groups` to `encoder_attention_groups` / `decoder_attention_groups`.
- Renamed the Voxel Mamba `d_model` to `embed_dim`, the OctFormer `num_blocks` to `encoder_depths` (and segmentation `channels` to `encoder_channels`), and the SPFormer U-Net `layers` to `depths`.
- Renamed the SphereFormer `base_channels`, `layers` and `block_reps` to `stem_channels`, `channels` and `depth`.
- Split the SparseUNet `channels` and `layers` into `encoder_channels`, `decoder_channels`, `encoder_depths` and `decoder_depths`, and renamed `base_channels` to `stem_channels`.
- Split the `segmentation` task into `semantic-segmentation` and `part-segmentation`, and renamed the `base` task to `pretraining`.
- Renamed `SegmentationModel` to `SemanticSegmentationModel` and `BaseModel` to `PretrainingModel`, and added `PartSegmentationModel`.
- Renamed the 3DETR classes from `DETR3D*` to `ThreeDETR*`, in the `threedetr` modules.
- Renamed the pretraining models to `<Arch>Pretraining`, and the Point-MAE and Point-M2AE part segmentation models to `*PartSegmentation`.
- Replaced `LitSegmentationModel` with `LitSemanticSegmentationModel` and `LitPartSegmentationModel`.
- Replaced the `generator` argument of the random transforms and `MixDataset` with `seed`, and added `set_random_state`.
- Fixed seeded random transforms repeating the same draws in every `DataLoader` worker (`seed_worker`).
- Updated PointNet++, VoteNet, PointRCNN, 3DETR, PVCNN++, Point-MAE and Point-M2AE to use the `PointNet2*` layers.
- Removed `SAModule`, `GlobalSAModule` and `FPModule`, replaced by `PointNet2SetAbstraction`, `PointNet2GlobalSetAbstraction` and `PointNet2FeaturePropagation`.
- Renamed the `pool` argument of `PointNet2Encoder` to `aggr` and of the PointNet++ models to `sa_aggr`.
- Updated the 17 checkpoints built on set abstraction to the new parameter names.
- Removed `utils.diffusion` and `utils.ensemble`, which nothing used.
- Removed `config.RANDOM_SEED`, which nothing read.

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
