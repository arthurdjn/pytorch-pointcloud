# Inferers

Inferers run a model over a point cloud at inference time. Several of them split a large cloud into smaller chunks, to keep inference efficient and within memory.

```{.python notest}
from torch_pointcloud.inferers import SlidingWindowInferer, TTAInferer
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate

inferer = TTAInferer(
    base=SlidingWindowInferer(block_size=6.0),
    transforms=Compose([
        RandomRotate(keys="pos", angle_range=(-180.0, 180.0), axis=2, p=1.0),
        RandomFlip(keys="pos", axes=[0, 1], p=0.5),
    ]),
    num_passes=4,
    aggregate="mean",
)

data = {"pos": torch.randn(1000, 3), "x": torch.randn(1000, 3), "batch": torch.zeros(1000)}
probs = inferer(data, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
```

## Which one to use

| Inferer                                                         | Runs the predictor on                                 | Reproduces                                             |
| --------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| [`SimpleInferer`](../api/inferers/simple.md)                    | the whole scene, once                                 | single-pass evaluation, the Lightning default          |
| [`SlidingWindowInferer`](../api/inferers/sliding_window.md)     | cubic blocks on a regular grid                        | block-based S3DIS / ScanNet protocols                  |
| [`VoxelPartitionInferer`](../api/inferers/voxel_partition.md)   | whole-extent downsamples, one point per voxel         | the fragment protocol of sparse and point transformers |
| [`KNNWindowInferer`](../api/inferers/knn_window.md)             | fixed-budget kNN crops around the least-covered point | possibility-driven crop voting (RandLA-Net)            |
| [`PotentialSphereInferer`](../api/inferers/potential_sphere.md) | radius spheres drawn from a potential grid            | potential sphere voting (KPConv)                       |
| [`TTAInferer`](../api/inferers/tta.md)                          | any base inferer, under several views                 | test-time augmentation and voting                      |
| [`PartRefinementInferer`](../api/inferers/part_refinement.md)   | any base inferer, then a neighbor vote                | part-segmentation label refinement                     |

The last two wrap another inferer rather than replacing it.
