# FAQ

## Installation

### `ImportError: Optional module 'spconv.pytorch' is not installed`

The model you built needs a kernel package that is not part of `pip install torch-pointcloud`. The message names the
module; the [installation](installation.md) table maps each package to the families that need it, and the selector
writes the `pip` line for your torch and CUDA versions. The rest of the library imports and runs without it.

### `pyg-lib` errors on a point-based model

Farthest point sampling, $k$-NN and radius search go through :pyg: [`torch-geometric`](https://pytorch-geometric.readthedocs.io/),
which runs them on `pyg-lib`. Install the wheel built for your torch version:

```bash
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

### A kernel package fails to import after a torch upgrade

Every optional package is compiled against one torch and CUDA build. Check `python -c "import torch; print(torch.__version__)"`
and reinstall the package from the matching index; the [compatibility](installation.md#compatibility) notes list the
versions the tests and benchmarks run on.

### Which models run on CPU?

The point-based families (PointNet, PointNet++, DGCNN, PointNeXt, PointMLP, KPConv, RandLA-Net, PointConv,
Point Transformer V1 and V2, the masked-point-modeling backbones). The sparse-voxel and flash-attention families (Point
Transformer V3, Sonata, Concerto, Utonia, SpUNet, SPFormer-UNet, SPVCNN, OctFormer, SphereFormer, the voxel-based
detectors, PointMamba) need a CUDA device.

## Pretrained weights

### Where are the weights stored?

Under `~/.cache/torch-pointcloud/models/<repo>/<revision>/<file>`, mirroring the Hub path of the registered URL.
`TORCH_POINTCLOUD_MODELS_DIR` moves the models cache, `TORCH_POINTCLOUD_CACHE_DIR` the whole cache. A checkpoint
pinned to a new revision is downloaded again.

### Can I run offline?

Yes, once the files are in the cache: load each model once on a connected machine, or copy its cache directory. The
weights are fetched from `https://huggingface.co` (the `HF_ENDPOINT` constant of `torch_pointcloud.models._registry`,
not an environment variable). A checkpoint you hold elsewhere loads with `create_model(name, checkpoint_path="...")`.

### `ValueError: No pretrained weights are registered for model 'pointnet.modelnet40'`

A name without an author tag (`pointnet.modelnet40`, `pointnext-sm`) is a configuration: architecture
hyperparameters and a transform, no weights. Build it with `pretrained=False` (the default) or list the names that
ship weights:

```{.python notest}
import torch_pointcloud as tp

tp.list_models(task="classification", pretrained=True)
```

### What license do the weights have?

Each checkpoint keeps the license of its source, readable without building the model:

```{.python notest}
from torch_pointcloud.models import model_info

model_info("sonata-base.pretrain.fair")["weights"]["license"]  # 'CC-BY-NC-4.0'
```

Sonata, Concerto and Utonia are CC-BY-NC-4.0 and cannot be used commercially; the others are MIT or Apache-2.0.
[`THIRD_PARTY_NOTICES.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/THIRD_PARTY_NOTICES.md) lists
every source.

## Models

### How do I get features instead of logits?

Build with `num_classes=0`: `forward` returns the representation under the head, $(B, C)$ for a classifier and $(N, C)$
for a segmentation model, and `model.num_features` is $C$. On a model that keeps its head,
`forward_head(..., pre_logits=True)` returns the same tensor and `forward_features` runs the encoder alone. The
[feature maps](models/features.md) page shows both on real scenes.

```{.python notest}
import torch_pointcloud as tp

encoder = tp.create_model("pointnet2-ssg.modelnet40.xu-yan", task="classification", pretrained=True, num_classes=0)
encoder.num_features  # 1024
```

### How do I reuse a pretrained encoder with my own head?

Pass your class count: `create_model(name, pretrained=True, num_classes=K)` loads the encoder weights and skips the
head's, with a warning listing the checkpoint keys it ignored. `reset_classifier(K)` does the same on a built model.
For a head of your own, build with `num_classes=0` and put the model in your `nn.Module`.

### `TypeError: Model 'pointnext-sm' registers architecture hparams only`

The configuration does not know your data: pass `in_channels=` and `num_classes=` to `create_model`.

### How do I run a model on a scene that does not fit in memory?

With an [inferer](inferers/overview.md): `SlidingWindowInferer` and `KNNWindowInferer` predict blocks,
`VoxelPartitionInferer` one point per voxel per pass, `PotentialSphereInferer` spheres until every point is covered.
Each returns one prediction per input point.

## Data

### My tensors are $(B, N, 3)$. What does the model expect?

Packed batches: points concatenated to $(N, 3)$ with a `batch` index $(N,)$. `collate` builds them from a list of
sample dicts; the [data format](data-format.md) page is the contract.

### Which value marks unlabeled points?

It depends on the checkpoint: the registered transform of a ScanNet or S3DIS model relabels everything outside its
class list to $-1$, the SemanticKITTI and nuScenes models and the DGCNN ScanNet model use $255$. Pass the same value
as `ignore_index` to the loss and to `confusion_matrix`. See [ignore index](data-format.md#ignore-index).

### S3DIS, ScanNet or ScanObjectNN will not download

They are released under terms of use. Read them (each class links its `terms_url`), then pass `accept_terms=True`
next to `download=True`.

### How do I make a run reproducible?

`torch_pointcloud.utils.random.seed_everything(seed)` seeds Python, NumPy and torch, and `set_determinism(tf32=False)`
turns off TensorFloat-32 in fp32 matmuls and cuDNN convolutions, which shifts metrics against references measured
without it; the benchmarks call both. Deterministic kernel selection stays with `torch.use_deterministic_algorithms`.
`PointCloudDataLoader` re-seeds its workers, so random transforms draw different numbers in every worker and epoch.

## Project

### How do I cite the library?

With the entry on the [about](about.md#citation) page.

### Where do I report a bug or ask a question?

In the [issue tracker](https://github.com/arthurdjn/pytorch-pointcloud/issues).
[`CONTRIBUTING.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CONTRIBUTING.md) explains how to send a
fix.
