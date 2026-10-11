# Data format

A sample is a `dict` of tensors keyed by the members of `DataKeys` (`torch_pointcloud.utils.data`), a `StrEnum`
whose members are plain strings: `data[DataKeys.POS]` and `data["pos"]` address the same entry. Datasets emit samples,
transforms rewrite them, `collate` packs a list of them into one batch, and a model's `forward` reads the batch keys
listed in its `info["input_keys"]`.

## Packed batches

Point clouds are **packed**, as in :pyg: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): the $B$ clouds
of a batch are concatenated along the first axis and a `batch` index tags each point with its cloud. Nothing is padded
to a common size, so for clouds of $N_1, \ldots, N_B$ points the batch holds $N = N_1 + \cdots + N_B$ rows, and
`pos[batch == i]` is cloud $i$.

![Three clouds of different sizes as a list of tensors, as one padded tensor, and packed](assets/animations/batch_modes.webp)

Models take packed tensors positionally, `model(x, pos, batch)` for most of them, and never a $(B, N, \cdot)$ tensor.

## Keys

Per-point tensors have $N$ rows in a sample and, after `collate`, $N$ rows over the whole batch.

| Key         | Shape                | Dtype            | Meaning                                                                                                                          |
| ----------- | -------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `pos`       | $(N, 3)$             | float32          | Coordinates                                                                                                                      |
| `x`         | $(N, C)$             | float32          | Model input features, assembled by the checkpoint's transform (color, normals, height); `None` for a model with `in_channels=0` |
| `batch`     | $(N,)$               | int64            | Cloud index of each point, $0 \ldots B-1$; written by `collate`                                                                  |
| `color`     | $(N, 3)$             | uint8 or float32 | RGB in $[0, 255]$ or $[0, 1]$, per dataset; checkpoint transforms divide by 255 where needed                                    |
| `normal`    | $(N, 3)$             | float32          | Surface normals                                                                                                                  |
| `intensity` | $(N, 1)$             | float32          | LiDAR return                                                                                                                     |
| `segment`   | $(N,)$               | int64            | Semantic label; see [ignore index](#ignore-index)                                                                                |
| `instance`  | $(N,)$               | int64            | Instance id, $-1$ for points in no instance                                                                                      |
| `pos_grid`  | $(N, 3)$             | int64            | Integer voxel coordinates, written by `Voxelize(dst_pos_grid_key=...)`; the sparse-voxel models take it in place of `pos`         |
| `face`      | $(F, 3)$             | int64            | Mesh triangles (ModelNet), sampled into points by `RandomSampleFaceVertices`                                                      |

Per-sample tensors have no point axis. `collate` stacks a 0-d tensor to $(B,)$ and concatenates anything else along
its first axis, so a per-sample vector keeps its own axis only when listed in `stack_keys`.

| Key        | Shape                | Dtype | Meaning                                                                                                              |
| ---------- | -------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
| `label`    | scalar, $(B,)$ batched | int64 | Class of the cloud (classification)                                                                                |
| `category` | scalar               | int64 | Object category (part segmentation); the checkpoint transform one-hots it to $(1, 16)$, which `collate` concatenates to $(B, 16)$ |
| `name`     | str                  |       | Sample identity, collated to a list                                                                                  |

Detection samples carry $K$ boxes, a different count per scene:

| Key         | Shape    | Dtype   | Meaning                                                                                                                            |
| ----------- | -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `box`       | $(K, 7)$ | float32 | $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$: center, full extents, heading                                                            |
| `label`     | $(K,)$   | int64   | Class of each box                                                                                                                  |
| `velocity`  | $(K, 2)$ | float32 | BEV velocity $(v_x, v_y)$ (nuScenes); a training pipeline appends it to `box` with `Cat`, and the $(K, 9)$ box carries it through the augmentations, the anchor targets and the losses |
| `calib`     | $(3, 4)$ | float32 | LiDAR-to-image matrix (KITTI, `return_calib=True`), read by the KITTI metric                                                      |
| `batch_box` | $(K,)$   | int64   | Scene index of each box, written by `collate` for the keys in `cat_keys`                                                           |

The voxel and pillar detectors read the voxel grid instead of the points: `voxel` $(V, P, C)$, `pos_voxel` $(V, 3)$ and
`voxel_num_points` $(V,)$ from `HardVoxelize`, with `batch_pos_voxel` from `cat_keys`. OctFormer reads an `octree`
built by `BuildOctree`. `info["input_keys"]` names the keys of each checkpoint, and `select_inputs` gathers them:

```{.python notest}
import torch
import torch_pointcloud as tp
from torch_pointcloud.utils.data import collate, select_inputs

model, info = tp.create_model("pointnet.modelnet40", return_info=True)
batch = collate([info["transform"]({"pos": torch.rand(2048, 3)})])
logits = model(*select_inputs(batch, info["input_keys"]))
```

Model outputs follow the same layout: $(B, \text{num\_classes})$ logits for a classifier, $(N, \text{num\_classes})$
for segmentation, and for a detector a `Detection3D` dict (`boxes`, `labels`, `scores`, `batch`, and `velocity` when
the head regresses it) from `decode`, packed over the batch like the ground truth.

## Collation

`collate` turns a list of samples into one batch. Every key must be present in every sample.

- tensors with one or more axes and the same trailing shape are concatenated along axis 0; 0-d tensors and Python
  scalars are stacked to $(B,)$; strings and other objects are collected in a list;
- `batch` $(N,)$ is synthesized from the leading size of `batch_from` (`pos` by default) under `batch_key`;
- `stack_keys` are stacked to a new leading axis, $(M, \cdot) \to (B, M, \cdot)$, for per-scene tensors of a fixed
  size;
- `cat_keys` stay concatenated and additionally get a `batch_<key>` scene index, for ragged per-scene ground truth
  such as `box`.

```{.python notest}
from torch_pointcloud.utils.data import PointCloudDataLoader

dataloader = PointCloudDataLoader(dataset, batch_size=8, cat_keys=["box", "label"])
data = next(iter(dataloader))
data["box"].shape, data["batch_box"].shape  # (K, 7), (K,)
```

`PointCloudDataLoader` is a `DataLoader` whose `collate_fn` is `collate` with these arguments; datasets never declare
them, because a transform may add or derive the keys that must stack or cat. It also re-seeds its workers, so seeded
random transforms draw different numbers in every worker and epoch, and takes `mix` / `mix_prob` to merge pairs of
samples before collation (`Mix3D`).

## Sampling keys

A transform that changes the number of points keeps `pos`, `x`, `segment` and `batch` aligned at the new resolution
and can record how to get back to the source cloud. The maps are opt-in: the registered pipelines set them, and a
sampler of your own writes them when given the `dst_*` keys.

| Key              | Shape                  | Written by                                                                                            | Meaning                                                                     |
| ---------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `origin_pos`     | $(N_\text{origin}, 3)$ | a `CopyItems` step at the head of the pipeline                                                        | the source cloud, in the same frame as `pos`                                |
| `origin_segment` | $(N_\text{origin},)$   | the same step, when the pipeline carries labels                                                       | the source labels, in the model's label space                               |
| `inverse`        | $(N_\text{origin},)$   | `Voxelize`, `DivisiblePad` through `dst_inverse_key`                                                  | source row to predictor row: `preds[inverse]` scores the full resolution    |
| `index`          | $(N,)$                 | the selection samplers (`FarthestPointSample`, `RandomSample`, `SphereCrop`, ...) through `dst_index_key` | predictor row to source row: `origin_pos[index]` is `pos`                 |

Chained steps compose these maps, so they always address the outermost source. After `collate` a map still indexes
the rows of its own sample: pass it in `cat_keys` and shift it with `offset_index(index, batch_index, batch)` to
address the packed batch.

## Ignore index

Raw datasets keep their own convention: unlabeled points are $0$ in ScanNet (`<unk>`) and in the outdoor sets
(`SEMANTIC3D_IGNORE_IDX`, `TORONTO3D_IGNORE_IDX`, `PARISLILLE3D_IGNORE_IDX`), S3DIS has none, and ScanNet instances
use $-1$. A checkpoint's registered transform remaps `segment` with `Relabel` and sends everything outside its class
list to the value its training used: $-1$ for the ScanNet and S3DIS checkpoints (Point Transformer V1 to V3, SpUNet,
SPFormer-UNet, Sonata, Concerto, Utonia), $255$ for the SemanticKITTI and nuScenes checkpoints (RandLA-Net, SPVCNN,
SphereFormer) and for the DGCNN ScanNet model. `Relabel` itself defaults to $0$. Pass the checkpoint's value as
`ignore_index` to the loss (`nn.CrossEntropyLoss(ignore_index=-1)`) and to `confusion_matrix`.
