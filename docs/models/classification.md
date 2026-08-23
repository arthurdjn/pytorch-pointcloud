# Classification

Classification maps **one point cloud to one label**: a chair, a table, a plane. Models read a packed batch $(N, C)$ and return logits $(B, \text{num\_classes})$, one row per cloud.

![Six committed sample objects, each captioned with the class a pretrained classifier gives it](../assets/tasks/classification.png)

## Run a pretrained checkpoint

Download the [`sample.ply`](../assets/data/sample.ply) (90 kB) to get started.

```bash
curl -LO https://github.com/arthurdjn/pytorch-pointcloud/raw/main/docs/assets/data/sample.ply
```

The function `create_model(..., return_info=True)` returns the network *and* its registry entry.

```{.python notest}
import numpy as np
import torch
from plyfile import PlyData

import torch_pointcloud as tp
from torch_pointcloud.utils.data import collate

# Load the pretrained model
model, info = tp.create_model(
    "pointnet2-ssg.modelnet40.xu-yan",
    task="classification",
    pretrained=True,
    return_info=True,
)
model = model.eval()

# Get associated transform
transform = info["transform"]

# Load the sample point cloud
ply = PlyData.read("sample.ply")["vertex"]
pos = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype("float32")

sample = {"pos": torch.from_numpy(pos)}
sample = transform(sample)

# Collate the sample into a batch
data = collate([sample])

# Inference pass
with torch.no_grad():
    logits = model(data.get("x"), data["pos"], data["batch"])  # (1, 40)

# Get predicted classes
classes = info["weights"]["classes"]
top = logits.softmax(dim=-1).topk(3, dim=-1)
for index, score in zip(top.indices[0].tolist(), top.values[0].tolist()):
    print(f"{classes[index]:>12}  {score:.2f}")
```

```text
    airplane  1.00
       plant  0.00
      stairs  0.00
```

!!! tip "Use `data.get("x")` rather than `data["x"]`"
    Checkpoints trained on coordinates only never build an `x` key, and the models accept `None`.

## Inputs and outputs

| Argument    | Shape                      | Meaning                                                                              |
| ----------- | -------------------------- | ------------------------------------------------------------------------------------ |
| `x`         | $(N, C)$ or `None`         | Per-point features, assembled by the checkpoint's transform (normals, height, color) |
| `pos`       | $(N, 3)$                   | Coordinates, all clouds in the batch concatenated                                    |
| `batch`     | $(N,)$                     | Per-point cloud index, $0 \ldots B-1$                                                |
| **returns** | $(B, \text{num\_classes})$ | Class logits, one row per cloud                                                      |

## Evaluate on a dataset

The same transform slots into a dataset, so reproducing a published score is a plain loop. `ModelNetNormalResampled` is the presampled ModelNet40 release the published numbers use; it downloads on first use and takes a few GB on disk.

```{.python notest}
import torch
import torch_pointcloud as tp
from torch_pointcloud.datasets import ModelNetNormalResampled
from torch_pointcloud.utils.data import PointCloudDataLoader

model, info = tp.create_model(
    "pointnet2-ssg.modelnet40.xu-yan",
    task="classification",
    pretrained=True,
    return_info=True,
)
model = model.cuda().eval()

dataset = ModelNetNormalResampled(
    root="data", 
    variant="40", 
    train=False,
    transform=info["transform"],
)
dataloader = PointCloudDataLoader(dataset, batch_size=32, num_workers=6)

correct = total = 0
with torch.no_grad():
    for data in dataloader:
        logits = model(None, data["pos"].cuda(), data["batch"].cuda())
        correct += (logits.argmax(dim=-1).cpu() == data["label"]).sum().item()
        total += data["label"].numel()

print(f"overall accuracy: {correct / total:.4f}")  # 0.9230
```

## Train from scratch

Build an untrained model by name and pass the data-dependent arguments. `in_channels` counts the columns of `x` (0 when the model reads coordinates only).

```{.python notest}
import torch
import torch_pointcloud as tp

model = tp.create_model(
    "pointnet2-ssg.modelnet40.xu-yan",
    task="classification",
    in_channels=0,
    num_classes=12,
)

pos = torch.randn(4096, 3)
batch = torch.arange(4).repeat_interleave(1024)
logits = model(None, pos, batch)
print(tuple(logits.shape))
```

```text
(4, 12)
```

## Fine-tune a pretrained backbone

You can fine-tune a pretrained backbone by overriding `num_classes` while loading pretrained weights.
This will load the backbone and rebuild the head with the desired number of classes.

```{.python notest}
model = tp.create_model(
    "pointnet2-ssg.modelnet40.xu-yan",
    task="classification",
    pretrained=True,
    num_classes=10,
)

# OR
model.reset_classifier(10)
```

```text
UserWarning: Skipping checkpoint keys ... with mismatched shapes, keeping their
initialization: head.lins.2.bias, head.lins.2.weight.
```

`model.reset_classifier(num_classes)` swaps the head on an already-built model, and `num_classes=0` replaces it with `nn.Identity` so the model returns the pooled descriptor instead of logits.
