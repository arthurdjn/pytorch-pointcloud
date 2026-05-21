# Inferers

`torch-pointcloud.inferers` is the test-time inference layer. It mirrors [MONAI's `monai.inferers`](https://docs.monai.io/en/stable/inferers.html): one `Inferer` ABC, several concrete strategies, and a wrap-and-compose pattern so that test-time augmentation (TTA) layers on top of any base inferer.

```python
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
