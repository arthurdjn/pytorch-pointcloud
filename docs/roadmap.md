# Roadmap

Where the library is going. The roadmap is directional: no dates and no fixed order, and items move as the work
shows what matters.

## Status

**Available now**

- [x] **Model zoo.** Every registered architecture and its checkpoints behind one `create_model` call, each with its
  transform, metrics and license.
- [x] **Packed batches.** The PyG layout end to end: datasets, transforms, models, inferers, losses and metrics.
- [x] **Inferers.** Sliding windows, voxel partitions, potential spheres and test-time augmentation for scenes of any
  size.
- [x] **Examples.** One plain PyTorch script per model and task: the authors' recipe to train, the reference protocol to
  benchmark.
- [x] **Lightning.** Optional modules for classification, segmentation and detection.

**In development**

- [ ] **Training recipes.** A training script for every checkpoint of the zoo, measured against the reference.
- [ ] **Reference protocols.** Every benchmark score reproduced with the authors' tool before it is published.

**Up next**

- [ ] **One encoder and head API.** The same `forward_features` and `forward_head` signatures across every
  family, so any backbone pairs with any task head.
- [ ] **Faster kernels.** Neighbor search, farthest point sampling and voxelization, on GPU and CPU.
- [ ] **More datasets and checkpoints.** Each with its reference protocol reproduced and measured.

## Models and weights

- [ ] Uniform `forward_features` and `forward_head` across the families.
- [ ] Coordinates-only input (`x=None`) on every point-based model.
- [ ] More checkpoints, converted and measured against the authors' numbers.

## Data and transforms

- [ ] A seed and a random state on transforms, in place of the `generator` parameter.
- [ ] An explicit list of the keys a transform may skip.
- [ ] The inference transform applied inside the inferer.

## Training and speed

- [ ] A training recipe for every checkpoint.
- [ ] Experiment configs for every model, as an optional extra.
- [ ] Faster neighbor search, sampling and voxelization.
- [ ] Fewer required dependencies: `plyfile` becomes optional.
