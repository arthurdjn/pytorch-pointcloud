# Datasets

`torch-pointcloud` ships loaders for the standard point-cloud benchmarks. Each loader returns a single-scene `dict` (the format consumed by [transforms](../transforms/overview.md)) and integrates with `torch.utils.data.DataLoader` via the `collate` helper in `torch_pointcloud.utils.data`.

```{.python notest}
from torch.utils.data import DataLoader
from torch_pointcloud.datasets import ModelNet40
from torch_pointcloud.utils.data import collate

dataset = ModelNet40(root="data", split="train")
loader = DataLoader(dataset, batch_size=32, collate_fn=collate)
```

## Cheat sheet

### Object classification

| Dataset | Task | Samples | Classes | License | API |
| --- | --- | --- | --- | --- | --- |
| **ModelNet10 / ModelNet40** | Object classification | ~12k | 10 / 40 | Research | [`modelnet`](../api/datasets/modelnet.md) |
| **ModelNet40Hdf5** | Object classification (pre-sampled 2,048 points + normals) | ~12k | 40 | Research | [`modelnet`](../api/datasets/modelnet.md) |
| **ShapeNetPart** | Part segmentation | ~16k | 16 categories / 50 parts | Research | [`shapenetpart`](../api/datasets/shapenetpart.md) |
| **ScanObjectNN** | Real-world object classification | 2.9k | 15 | Research | [`scanobjectnn`](../api/datasets/scanobjectnn.md) |

### Indoor scene segmentation

| Dataset | Task | Scenes | Classes | License | API |
| --- | --- | --- | --- | --- | --- |
| **S3DIS** | Indoor semantic segmentation | 271 rooms, 6 areas | 13 | Research | [`s3dis`](../api/datasets/s3dis.md) |
| **ScanNet v2** | Indoor semantic + instance | 1.5k scenes | 20 (NYU40) / 200 | Research | [`scannet`](../api/datasets/scannet.md) |

### Outdoor / driving segmentation

| Dataset | Task | Frames | Classes | License | API |
| --- | --- | --- | --- | --- | --- |
| **SemanticKITTI** | LiDAR semantic segmentation | 43k frames | 19 | Research | [`semantickitti`](../api/datasets/semantickitti.md) |
| **Semantic3D** | Outdoor terrestrial laser scans | 30 scenes | 8 | Research | [`semantic3d`](../api/datasets/semantic3d.md) |
| **Toronto3D** | Mobile mapping (street scenes) | 4 areas | 8 | Research | [`toronto3d`](../api/datasets/toronto3d.md) |
| **ParisLille3D** | Mobile mapping (streets) | 3 scenes | 9 | Research | [`parislille3d`](../api/datasets/parislille3d.md) |

### Base class

| Dataset | Task | Notes | API |
| --- | --- | --- | --- |
| **PointCloudDataset** | (any) | Abstract base class all loaders build on: `raw/` + `processed/` disk layout, `download` / `process` hooks. Subclass it for custom data. | [`pointcloud`](../api/datasets/pointcloud.md) |

## Dict keys

All datasets emit dicts using the standard key conventions from `DataKeys` in `torch_pointcloud.utils.data`:

| Key | Shape | Description |
| --- | --- | --- |
| `pos` | $(N, 3)$ | 3D coordinates |
| `color` | $(N, 3)$ | RGB (uint8 or float, depending on dataset) |
| `normal` | $(N, 3)$ | Surface normals (when available) |
| `segment` | $(N,)$ | Semantic labels (segmentation datasets) |
| `instance` | $(N,)$ | Instance IDs (ScanNet) |
| `label` | scalar | Object class (classification datasets) |
| `face` | $(F, 3)$ | Triangle indices (ModelNet / mesh datasets) |

After `collate`, per-point tensors are concatenated along axis 0 and a `batch` key of shape $(N,)$ is appended to identify each point's source scene.

!!! warning "Color and ignore-index conventions vary per dataset"
    `color` is uint8 in $[0, 255]$ for most indoor loaders (e.g. `S3DIS`), float32 in $[0, 255]$ for `ScanNet`, and float32 in $[0, 1]$ for `S3DISHdf5`; check the per-dataset API page before normalizing. Unlabeled points use label 0 (`<unk>` / outdoor conventions, e.g. `ScanNet`), -1 (indoor no-instance and class-subset remaps, e.g. `S3DIS`), or 255 (the `SemanticKITTI` remap example).

## Picking a dataset

- **Sanity-check classification**: `ModelNet10` (small, downloads fast).
- **Modern classification benchmark**: `ScanObjectNN` (real-world scans, harder).
- **Indoor segmentation reference**: `S3DIS` (small) or `ScanNet` (larger).
- **Driving / LiDAR**: `SemanticKITTI`.
- **Custom data**: subclass `PointCloudDataset`.

## Downloading

Most loaders auto-download to `root=...` on first use; some (S3DIS, SemanticKITTI) require manual download due to license clickwraps. See the per-dataset API page for the exact URL and license.
