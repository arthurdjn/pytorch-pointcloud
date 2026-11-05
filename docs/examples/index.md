---
title: Tutorials
---

# Tutorials

Runnable notebooks covering the library end to end. Open them in Colab from the badge at the top of each page.

## Beginner

The scene tutorial reads a ScanNet scan, which requires accepting the ScanNet terms of use.

<div class="grid cards tutorial-cards" markdown>

-   :material-rocket-launch: __[Quickstart: classify a point cloud](01-quickstart.md)__

    ![A point cloud object classified by a pretrained model](../assets/tutorials/thumbs/01-quickstart.png)

    Load a pretrained model, build a transform pipeline, and classify an object in a few lines.

-   :material-domain: __[Segment a scene](02-segmentation-inference.md)__

    ![An indoor room with every point colored by its predicted class](../assets/tutorials/thumbs/02-segmentation-inference.png)

    Run per-point semantic segmentation on a full indoor scene with a tiling inferer.

-   :material-tune: __[Preprocessing pipelines](03-transforms.md)__

    ![The same cloud before and after a transform pipeline](../assets/tutorials/thumbs/03-transforms.png)

    Compose dict transforms for sampling, normalization, and augmentation, and inspect each step.

</div>

## Intermediate

Your own data and a plain PyTorch training loop.

<div class="grid cards tutorial-cards" markdown>

-   :material-database-plus: __[Use your own data](04-custom-dataset.md)__

    ![Several clouds collated into one packed batch, one color per sample](../assets/tutorials/thumbs/04-custom-dataset.png)

    Wrap your own point cloud files in a `Dataset`, attach a transform, and collate them into packed batches.

-   :material-school: __[Train a model](05-training.md)__

    ![A training curve falling over successive epochs](../assets/tutorials/thumbs/05-training.png)

    Train a classification model from scratch: dataloaders, optimizer, and the evaluation loop.

-   :material-magnify-scan: __[Features and similarity search](06-feature-search.md)__

    ![A scene colored by the principal components of a pretrained encoder's features](../assets/tutorials/thumbs/06-feature-search.png)

    Read a frozen encoder's features, query one point, and retrieve whole shapes by descriptor.

</div>

## Advanced

3D object detection on driving LiDAR.

<div class="grid cards tutorial-cards" markdown>

-   :material-car: __[Detect objects in driving LiDAR](07-driving-detection.md)__

    ![A driving LiDAR sweep with oriented boxes around detected vehicles](../assets/tutorials/thumbs/07-driving-detection.png)

    Turn one sweep into oriented 3D boxes: voxel encoding, decoding, non-maximum suppression.

-   :material-terrain: __[Segment an aerial LiDAR survey](08-large-scale-lidar.md)__

    ![A city block of airborne LiDAR with every point colored by its predicted class](../assets/tutorials/thumbs/08-large-scale-lidar.png)

    Train on 1 km tiles, run a sliding window over a whole tile, and write the classes back into the LAZ.

-   :material-tooth-outline: __[Segment teeth on intraoral scans](09-dental-scans.md)__

    ![A dental arch with every tooth colored by its predicted label](../assets/tutorials/thumbs/09-dental-scans.png)

    Train on 1200 jaw scans, label every vertex of a test scan with its FDI tooth number, and read the confusions.

</div>
