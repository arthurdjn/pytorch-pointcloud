---
title: Tutorials
---

# Tutorials

Guided, runnable notebooks covering the library end to end. Each page is rendered from a Jupyter
notebook committed under `docs/examples/`; open it in Colab or download it from the badges at the
top of every tutorial.

<div class="grid cards" markdown>

-   :material-rocket-launch: __[Quickstart: classify a point cloud](01-quickstart.md)__

    Load a pretrained model, build a transform pipeline, and classify an object in a few lines.

-   :material-domain: __[Segment a scene](02-segmentation-inference.md)__

    Run per-point semantic segmentation on a full indoor scene with a tiling inferer.

-   :material-tune: __[Preprocessing pipelines](03-transforms.md)__

    Compose dict transforms for sampling, normalization, and augmentation, and inspect each step.

-   :material-database-plus: __[Use your own data](04-custom-dataset.md)__

    Wrap your own point cloud files in a dataset with disk caching and packed-batch collation.

-   :material-school: __[Train a model](05-training.md)__

    Train a segmentation model from scratch: dataloaders, optimizer, and the evaluation loop.

</div>
