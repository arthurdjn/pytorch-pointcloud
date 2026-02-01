# pytorch-pointcloud

PyTorch Point Cloud models, scripts, pretrained weights -- PointNet, PointNet++, DGCNN, KPConv, RandLA-Net, SPConv, VoteNet, PointGroup, SPVCNN, 3DETR, PointTransformer and more

## Installation

```bash
# If using torchsparse, you need google-sparsehash
sudo apt-get install libsparsehash-dev
# or from conda: 


uv venv --clear

# For CUDA specific, it is recommended to install the dependencies manually
# depending on your torch and CUDA version
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
# uv pip uninstall torch_scatter torch_cluster pyg_lib torch_spline_conv

uv pip install --no-cache-dir --no-binary :all: --no-build-isolation 'mamba-ssm[causal-conv1d]'   
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
uv pip install spconv-cu126 

# Install all extras (NOTE: Some extras are Linux and CPU only)
# uv sync --all-extras --dev

# Install the minimum dependencies
uv sync
# for some cuda specific packages like flash-attn, torchsparse
# uv pip install setuptools wheel rootpath
uv sync --extra build
```

### All in one

```bash
uv sync --all-extras --dev
uv pip uninstall torch torchvision torch_scatter torch_cluster pyg_lib torch_spline_conv
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 && uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html && uv pip install spconv-cu126
```
