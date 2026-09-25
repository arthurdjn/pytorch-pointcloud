# Quick Start

## Run a pretrained model

The example reads one object of the ModelNet40 test set, which
[`ModelNetNormalResampled`](api/datasets/modelnet.md) downloads on first use:

```{.python notest}
import torch

import torch_pointcloud as tp
from torch_pointcloud.datasets import ModelNetNormalResampled
from torch_pointcloud.utils.data import collate

# Instantiate the model and return associated info.
model, info = tp.create_model(
    "pointnet2-ssg.modelnet40.xu-yan",
    task="classification",
    pretrained=True,
    return_info=True,
)
model = model.eval()

# Get associated transform for inference.
transform = info["transform"]

# Load input data. The dataset applies the transform, which samples points and normals together.
dataset = ModelNetNormalResampled(root="data", variant="40", train=False, download=True, transform=transform)
sample = dataset[0]

# Pack into a batch. The provided `collate` function
# handles the packed-batch convention, but you can use your own.
batch = collate([sample])

# Forward pass. Models take packed batches, never padded (B, N, ...) tensors.
with torch.no_grad():
    logits = model(batch.get("x"), batch["pos"], batch["batch"])

classes = info["weights"]["classes"]
top = logits.softmax(dim=-1).topk(3, dim=-1)
for index, score in zip(top.indices[0].tolist(), top.values[0].tolist()):
    print(f"{classes[index]:>12}  {score:.2f}")
```

```text
    airplane  1.00
       plant  0.00
      guitar  0.00
```

![Six committed sample objects, each captioned with the class this checkpoint gives it](./assets/tasks/classification.png)

## Data conventions

Point clouds use the **packed** format of :pyg: PyTorch Geometric: instead of zero-padding clouds to a common size, they
are concatenated and each point is tagged with its sample index. For $B$ samples of $N_i$ points ($N = N_1 + \ldots + N_B$):

![Three clouds of different sizes as a list of tensors, as one padded tensor, and packed](./assets/animations/batch_modes.webp)

| Tensor  | Shape    | Description                              |
| ------- | -------- | ---------------------------------------- |
| `pos`   | $(N, 3)$ | 3D coordinates, all points concatenated  |
| `x`     | $(N, C)$ | Per-point features                       |
| `batch` | $(N,)$   | Per-point batch index $(0, \ldots, B-1)$ |

Datasets emit the other keys (`normal`, `color`, `segment`, `label`, ...), listed on the [Datasets](datasets/overview.md#dict-keys) page.
