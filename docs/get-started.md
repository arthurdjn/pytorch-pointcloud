# Get Started

## About

:pytorch-pointcloud: [`torch-pointcloud`](https://github.com/arthurdjn/pytorch-pointcloud){: target="_blank" } is a PyTorch library for deep learning on point clouds build on top of popular and powerful libraries such as :pytorch: [`torch`](https://pytorch.org){: target="_blank" } and :pyg: [`torch-geometric`](https://pytorch-geometric.readthedocs.io/en/latest/){: target="_blank" }. It implements a wide range of State-of-the-Art models for point cloud classification, segmentation, and other tasks.

## Installation

The :pytorch-pointcloud: `torch-pointcloud` package requires the :pytorch: `torch` package as a dependency.
To install it, run:

=== "pip"

    ```bash
    pip install torch_pointcloud
    ```

=== "uv"

    ```bash
    uv add torch-pointcloud
    ```

To install all extras, run:

=== "pip"

    ```bash
    pip install torch-pointcloud[all]
    ```
    
=== "uv"

    ```bash
    uv add torch-pointcloud[all]
    ```

## Contributing & Supporting

We welcome any contributions, from bug reports to new features! If you want to contribute to the package, please read the [For Developers](https://github.com/arthurdjn/pytorch-pointcloud#-for-developers) section.

If you simply find the package useful, please consider giving it a star ⭐️ on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/LICENSE) file for details.

| **Model**                          | **Description**                                        |
| :--------------------------------- | :----------------------------------------------------- |
| [`PointNet`](models/pointnet.md)   | PointNet is a simple point cloud classification model. |
| [PointNet++](models/pointnet++.md) | PointNet++ is a point cloud classification model.      |
| [DGCNN](models/dgcnn.md)           | DGCNN is a point cloud classification model.           |
| [KPConv](models/kpconv.md)         | KPConv is a point cloud classification model.          |
| [RandLA-Net](models/randlanet.md)  | RandLA-Net is a point cloud classification model.      |
| [SPVCNN](models/spvcnn.md)         | SPVCNN is a point cloud classification model.          |
| [VoteNet](models/votenet.md)       | VoteNet is a point cloud classification model.         |
| [PointGroup](models/pointgroup.md) | PointGroup is a point cloud classification model.      |
| [SPVCNN](models/spvcnn.md)         | SPVCNN is a point cloud classification model.          |
