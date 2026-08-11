# Installation

## Quick install

=== "uv"

    ```bash
    uv add torch-pointcloud
    ```

=== "pip"

    ```bash
    pip install torch-pointcloud
    ```

This installs the core package and pulls :pytorch: [`torch`](https://pytorch.org), :pyg: [`torch-geometric`](https://pytorch-geometric.readthedocs.io/), and a small set of mandatory dependencies.

## Optional dependencies

Several features depend on third-party CUDA extensions. Install them individually as you need them. The exact wheel URL changes with PyTorch / CUDA version, so pin both.

| Extra | Used by | Install |
| --- | --- | --- |
| `torch-scatter` | `Voxelize`, scatter-based pooling | `pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html` |
| `torch-cluster` | `FarthestPointSample`, kNN graph | `pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html` |
| `ocnn-pytorch` | `BuildOctree`, OctFormer | `pip install ocnn` |
| `torchsparse` | SPVCNN sparse-conv stems | `pip install torchsparse` |

Replace `${TORCH}` with your installed torch version (e.g. `2.5.0`) and `${CUDA}` with `cu121`, `cu124`, or `cpu`. Find your version with `python -c "import torch; print(torch.__version__)"`.

## Install everything

The `[all]` extra pulls every optional dependency:

=== "uv"

    ```bash
    uv add 'torch-pointcloud[all]'
    ```

=== "pip"

    ```bash
    pip install 'torch-pointcloud[all]'
    ```

!!! note
    `[all]` does not pin the CUDA-extension wheel URLs. If you need a specific CUDA build, install the optional packages individually with the `-f` flag shown above.

## From source (development)

```bash
git clone https://github.com/arthurdjn/pytorch-pointcloud.git
cd pytorch-pointcloud
uv sync --all-extras
```

Run the test suite to verify the install:

```bash
make test
```

## Compatibility

- **Python**: 3.10+
- **PyTorch**: 2.1+
- **CUDA**: optional; CPU-only inference and training work out of the box.

## Troubleshooting

- **`ImportError: No module named 'torch_scatter'`**: install the matching CUDA wheel from the PyG index above. The package will fall back to slower pure-Python paths if scatter is missing, but voxelization and some pooling layers will be unavailable.
- **`FileNotFoundError` when loading pretrained weights**: checkpoints are not published for automatic download yet. `pretrained=True` reads the weight file from the local cache (`~/.cache/torch-pointcloud/models/` by default, overridable with `TORCH_POINTCLOUD_MODELS_DIR`), so the file must already be there. Hub publication is pending.
- **OctFormer / SPVCNN crashes**: these models require `ocnn` and `torchsparse` respectively. Install them as shown above.
