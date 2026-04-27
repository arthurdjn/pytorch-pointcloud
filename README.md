# pytorch-pointcloud

PyTorch Point Cloud models, scripts, pretrained weights -- PointNet, PointNet++, DGCNN, KPConv, RandLA-Net, SPConv, VoteNet, PointGroup, SPVCNN, 3DETR, PointTransformer and more

## Installation

```bash
# If using torchsparse, you need google-sparsehash
sudo apt-get install libsparsehash-dev
# or from conda: 


uv venv --clear

# Install all extras (NOTE: Some extras are Linux and CPU only)
uv sync --all-extras --dev

uv pip uninstall torch torchvision torch-geometric torch_scatter torch_cluster pyg_lib torch_spline_conv

# For CUDA specific, it is recommended to install the dependencies manually
# depending on your torch and CUDA version
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

# Install pyg-lib
uv pip install pyg_lib torch-geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

# Install spconv
uv pip install spconv-cu126 

# Install mamba-ssm
uv pip install packaging wheel
uv pip install --no-cache-dir --no-build-isolation 'mamba-ssm[causal-conv1d]'

# DWConv
uv pip install dwconv

# Test
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run --no-sync python -c "import torch_cluster; print(torch_cluster.__version__)"
uv run --no-sync python -c "import torch_geometric; print(torch_geometric.__version__)"
uv run --no-sync python -c "import torch_pointcloud; print(torch_pointcloud.__version__)"
```

### All in one

```bash
uv sync --all-extras --dev
uv pip uninstall torch torchvision torch_scatter torch_cluster pyg_lib torch_spline_conv
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 && uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html && uv pip install spconv-cu126
```

### Setup Open3D + Torch-Pointcloud

It looks like Open3D-Ml supports only PyTorch 2.2.*
Annoying because open3d latest release is from 2025 and has no support (from PyPi) for latest pytorch release...

```bash
conda create -n open3d python=3.10

# 1. PyTorch 2.2.2 (pinned by Open3D's requirement)
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121

# 2. PyG companion wheels — MUST come from the PyG index matching your torch+cuda
pip install --no-cache-dir \
  pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 3. torch-geometric itself (no CUDA build, just a regular package)
pip install torch-geometric

# 4. Open3D last (won't touch torch if torch is already satisfied)
pip install open3d

# 5. Install torch-pointcloud
pip install -e .

# Install numpy<2 to avoid conflicts with torch-pointcloud
pip install "numpy<2"

# 6. Test
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_cluster; print(torch_cluster.__version__)"
python -c "import open3d.ml.torch as ml3d; print('ok')"
python -c "import torch_geometric; print(torch_geometric.__version__)"
python -c "import torch_pointcloud; print(torch_pointcloud.__version__)"
```
