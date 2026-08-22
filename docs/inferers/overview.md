# Inferers

:pytorch-pointcloud-mini: `torch-pointcloud.inferers` is the test-time inference layer. it contains an abstract base `Inferer`, several concrete strategies, and a wrap-and-compose pattern so that test-time augmentation (TTA) layers can be added on top of any base inferer.

```{.python notest}
from torch_pointcloud.inferers import SlidingWindowInferer, TTAInferer
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate

base = SlidingWindowInferer(block_size=6.0)
inferer = TTAInferer(
    base=base,
    transforms=Compose([
        RandomRotate(keys="pos", angle_range=(-180.0, 180.0), axis=2, p=1.0),
        RandomFlip(keys="pos", axes=[0, 1], p=0.5),
    ]),
    num_passes=4,
    aggregate="mean",
)
probs = inferer(data, predictor=lambda d: model(d["pos"], d["pos"], d["batch"]))
```

## Strategies

| Inferer                  | Runs the predictor on                                                                       | Reproduces                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `SimpleInferer`          | the whole scene, once                                                                       | single-pass evaluation (the Lightning default on test batches)           |
| `SlidingWindowInferer`   | axis-aligned blocks, blended by `mean`, `max` (most confident block) or `vote` (hard votes) | block-based S3DIS / ScanNet protocols scoring every point of the room    |
| `VoxelPartitionInferer`  | sub-clouds holding one point per voxel, so every raw point is predicted                     | the voxel-partition (fragment) protocol of sparse and point transformers |
| `KNNWindowInferer`       | k-nearest-neighbour crops around the least-covered point until coverage                     | possibility-driven crop voting (RandLA-Net)                              |
| `PotentialSphereInferer` | radius spheres drawn from a coarse potential grid, EMA of softmax                           | potential sphere voting (KPConv)                                         |
| `TTAInferer`             | any base inferer under enumerated or random views (`include_identity` adds a clean pass)    | test-time augmentation and voting                                        |
| `PartRefinementInferer`  | any base inferer, then a nearest-neighbour majority over rare / foreign part labels         | part-segmentation label refinement                                       |
