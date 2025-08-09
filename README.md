# pytorch-pointcloud

PyTorch Point Cloud models, scripts, pretrained weights -- PointNet, PointNet++, DGCNN, KPConv, RandLA-Net, SPConv, VoteNet, PointGroup, SPVCNN, 3DETR, PointTransformer and more

## Installation

```bash
# If using torchsparse, you need google-sparsehash
sudo apt-get install libsparsehash-dev
# or from conda: 

# For CUDA specific, it is recommended to install the dependencies manually
# depending on your torch and CUDA version
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
uv pip install spconv-cu124

# Install the minimum dependencies
uv sync
# for some cuda specific packages like flash-attn, torchsparse
# uv pip install setuptools wheel rootpath
uv sync --extra build

# Install all extras (NOTE: Some extras are Linux and CPU only)
uv sync --all-extras
```
