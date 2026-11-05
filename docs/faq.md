# FAQ

## Installation

??? faq "`ImportError: Optional module 'spconv.pytorch' is not installed`"

    The model needs a kernel package that `pip install torch-pointcloud` does not pull. The message names the module.
    The [installation](installation.md) table says which package each family needs, and the selector writes the
    `pip` line for your torch and CUDA versions. Everything else imports and runs without it.

??? faq "`pyg-lib` errors on a point-based model"

    Farthest point sampling, $k$-NN and radius search go through :pyg: [`torch-geometric`](https://pytorch-geometric.readthedocs.io/),
    which runs them on `pyg-lib`. Install the wheel built for your torch version:

    ```bash
    pip install pyg-lib -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
    ```

??? faq "A kernel package fails to import after a torch upgrade"

    Every optional package is compiled against one torch and CUDA build. Check
    `python -c "import torch; print(torch.__version__)"` and reinstall the package from the matching index. The
    [compatibility](installation.md#compatibility) notes list the versions the tests and benchmarks run on.

??? faq "Which models run on CPU?"

    The point-based families: PointNet, PointNet++, DGCNN, PointNeXt, PointMLP, KPConv, RandLA-Net, PointConv,
    Point Transformer V1 and V2, and the masked point modeling backbones. The sparse-voxel and flash-attention
    families need a CUDA device: Point Transformer V3, Sonata, Concerto, Utonia, SpUNet, SPFormer-UNet, SPFormer,
    ForAINet, SPVCNN, OctFormer, SphereFormer, PointMamba and the voxel-based detectors.

## Pretrained weights

??? faq "Where are the weights stored?"

    Under `~/.cache/torch-pointcloud/models/<repo>/<revision>/<file>`, the same path as on the Hub.
    `TORCH_POINTCLOUD_CACHE_DIR` moves the cache. A checkpoint
    pinned to a new revision is downloaded again.

??? faq "A download fails with HTTP 401 or 403"

    The repository is gated (`ptv3.dales.jayakumar-pujar`): accept its terms on its Hub page, then log in with
    `hf auth login` or set `HF_TOKEN`. The token is sent with the download.

??? faq "Can I run offline?"

    Yes, once the files are in the cache: load each model once on a connected machine, or copy its cache directory.
    The weights come from `https://huggingface.co` (the `HF_ENDPOINT` constant of `torch_pointcloud.utils.hub`,
    not an environment variable). A checkpoint stored somewhere else loads with
    `create_model(name, checkpoint_path="...")`.

??? faq "`ValueError: No pretrained weights are registered for model 'pointnet.modelnet40'`"

    A name without an author tag (`pointnet.modelnet40`, `pointnext-sm`) is a configuration: hyperparameters and a
    transform, no weights. Build it with `pretrained=False` (the default), or list the names that ship weights:

    ```{.python notest}
    import torch_pointcloud as tp

    tp.list_models(task="classification", pretrained=True)
    ```

??? faq "What license do the weights have?"

    Each checkpoint keeps the license of its source. You can read it without building the model:

    ```{.python notest}
    from torch_pointcloud.models import model_info

    model_info("sonata-base.pretrain.fair")["weights"]["license"]  # 'CC-BY-NC-4.0'
    ```

    Sonata, Concerto and Utonia are CC-BY-NC-4.0, so no commercial use. The others are MIT, Apache-2.0 or Etalab-2.0.
    [`THIRD_PARTY_NOTICES.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/THIRD_PARTY_NOTICES.md) lists
    every source.

## Models

??? faq "How do I get features instead of logits?"

    Build with `num_classes=0`. `forward` then returns the representation under the head, $(B, C)$ for a classifier
    and $(N, C)$ for a segmentation model, and `model.num_features` is $C$. On a model that keeps its head,
    `forward_head(..., pre_logits=True)` returns the same tensor and `forward_features` runs the encoder alone. The
    [feature maps](models/features.md) page shows both on real scenes.

    ```{.python notest}
    import torch_pointcloud as tp

    encoder = tp.create_model("pointnet2-ssg.modelnet40.xu-yan", task="classification", pretrained=True, num_classes=0)
    encoder.num_features  # 1024
    ```

??? faq "How do I reuse a pretrained encoder with my own head?"

    Pass your class count: `create_model(name, pretrained=True, num_classes=K)` loads the registered checkpoint and
    then rebuilds the head for `K` classes. `reset_classifier(K)` does the same on a built model. For a head of
    your own, build with `num_classes=0` and wrap the model in your `nn.Module`.

??? faq "`TypeError: Model 'pointnext-sm' registers architecture hparams only`"

    The configuration does not know your data. Pass `in_channels=` and `num_classes=` to `create_model`.

??? faq "How do I run a model on a scene that does not fit in memory?"

    With an [inferer](inferers/overview.md). `SlidingWindowInferer` and `KNNWindowInferer` predict blocks,
    `VoxelPartitionInferer` one point per voxel per pass, and `PotentialSphereInferer` spheres until every point is
    covered. Each returns one prediction per input point.

## Data

??? faq "My tensors are $(B, N, 3)$. What does the model expect?"

    Packed batches: the points concatenated to $(N, 3)$ with a `batch` index $(N,)$. `collate` builds them from a
    list of sample dicts. See the [data format](data-format.md) page.

??? faq "Which value marks unlabeled points?"

    It depends on the checkpoint. The registered transform of a ScanNet or S3DIS model relabels everything outside
    its class list to $-1$; the SemanticKITTI and nuScenes models and the DGCNN ScanNet model use $255$. Pass the
    same value as `ignore_index` to the loss and to `confusion_matrix`. See [ignore index](data-format.md#ignore-index).

??? faq "S3DIS, ScanNet or ScanObjectNN will not download"

    They are released under terms of use. Read them (each class links its `terms_url`), then pass `accept_terms=True`
    next to `download=True`.

??? faq "How do I make a run reproducible?"

    `torch_pointcloud.utils.random.seed_everything(seed)` seeds Python, NumPy and torch. `set_determinism(tf32=False)`
    turns off TensorFloat-32 in fp32 matmuls and cuDNN convolutions, which otherwise shifts metrics against references
    measured without it; the benchmarks call both. Deterministic kernel selection stays with
    `torch.use_deterministic_algorithms`. `PointCloudDataLoader` re-seeds its workers, so random transforms draw
    different numbers in every worker and epoch.

## Project

??? faq "How do I cite the library?"

    With the entry on the [about](about.md#citation) page.

??? faq "Where do I report a bug or ask a question?"

    In the [issue tracker](https://github.com/arthurdjn/pytorch-pointcloud/issues).
    [`CONTRIBUTING.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CONTRIBUTING.md) explains how to
    send a fix.
