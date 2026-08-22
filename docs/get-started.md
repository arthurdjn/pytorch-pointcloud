# Quick Start

This page takes you from zero to running a pretrained model on a point cloud in about fifteen lines.

## Install

Install the package with:

```bash
pip install torch-pointcloud
```

See [Installation](installation.md) for the CUDA extensions and optional features.

## Get Started

```{.python notest}
import torch
import torch_pointcloud as tp

# 1. Build a pretrained model. `create_model` looks up the architecture by
#    name and task, then loads its matching weights.
model = tp.create_model(
    "pointnext-sm.scanobjectnn.openpoints", 
    task="classification", 
    pretrained=True,
).eval()

# 2. A toy "scene" with 2048 random points. The ScanObjectNN weights expect
#    4 input channels: xyz plus a height feature.
pos = torch.randn(2048, 3)
x = torch.cat([pos, pos[:, 1:2] - pos[:, 1].min()], dim=1)
batch = torch.zeros(2048, dtype=torch.long)

# 3. Forward pass. Models accept packed batches: a flat (N, ...) tensor
#    plus a (N,) `batch` index, never padded (B, N, ...) tensors.
with torch.no_grad():
    logits = model(x, pos, batch=batch)

print(logits.shape)  # (1, num_classes)
```

The packed-batch convention comes from :pyg: PyTorch Geometric: instead of zero-padding clouds to a common size, we concatenate them and tag each point with its sample index. See [Data conventions](#data-conventions) below.

## Using a real dataset

```{.python notest}
import torch
from torch.utils.data import DataLoader
from torch_pointcloud.datasets import ModelNet10
from torch_pointcloud.utils.data import collate
import torch_pointcloud.transforms as T

transform = T.Compose([
    T.Rescale(keys="pos", method="centroid"),
    T.RandomSampleFaceVertices(
        keys="pos", face_key="face", normal_key="normal", num_samples=1024,
    ),
])

dataset = ModelNet10(root="data", train=False, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate)

for data in dataloader:
    pos, batch = data["pos"], data["batch"]
    x = torch.cat([pos, pos[:, 1:2] - pos[:, 1].min()], dim=1)
    logits = model(x, pos, batch=batch)
    preds = logits.argmax(dim=-1)
    break
```

`collate` understands the packed format: it concatenates per-point tensors along the batch axis and builds the `batch` index for you. Scene-level tensors (e.g. `label`) are stacked normally.

## Data conventions

All point clouds use a **packed (flat-batch) format**. For a batch of B samples with $N_i$ points each:

| Tensor            | Shape    | Meaning                                  |
| ----------------- | -------- | ---------------------------------------- |
| `pos`             | $(N, 3)$ | 3D coordinates, all points concatenated  |
| `x`               | $(N, C)$ | Per-point features                       |
| `batch`           | $(N,)$   | Per-point batch index $(0, \ldots, B-1)$ |
| `normal`, `color` | $(N, 3)$ | Per-point attributes                     |
| `segment`         | $(N,)$   | Per-point semantic label                 |
| `label`           | $(B,)$   | Scene / object-level label               |

$N = N_1 + N_2 + \ldots + N_B$.

## Next steps

<div class="grid cards" markdown>

-   :material-cube-outline: __[Models](models/overview.md)__: what's available, what each is good for
-   :material-database: __[Datasets](datasets/overview.md)__: built-in dataset loaders
-   :material-tune: __[Transforms](transforms/overview.md)__: composable preprocessing
-   :material-book-open-page-variant: __[API Reference](api/index.md)__: every public class and function

</div>
