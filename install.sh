#!/usr/bin/env bash
set -euo pipefail

# Optional system dep for torchsparse (Ubuntu/Debian):
# sudo apt-get install -y libsparsehash-dev

echo ">>> Creating a fresh virtual environment"
uv venv --clear

echo ">>> Installing all extras (NOTE: some extras are Linux and CPU only)"
uv sync --all-extras --dev

echo ">>> Uninstalling CPU wheels so we can replace them with CUDA builds"
# `|| true` since some packages may not have been installed.
uv pip uninstall torch torchvision torch-geometric torch_scatter torch_cluster pyg_lib torch_spline_conv || true

echo ">>> Installing torch + torchvision (CUDA 12.6)"
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

echo ">>> Installing pyg-lib and friends (CUDA wheels)"
uv pip install pyg_lib torch-geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

echo ">>> Installing spconv-cu126"
uv pip install spconv-cu126

echo ">>> Installing causal-conv1d + mamba-ssm (force from-source build, --no-deps so torch is left alone)"
uv pip uninstall mamba-ssm causal-conv1d || true
uv pip install packaging wheel
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --no-deps --no-cache-dir --no-build-isolation causal-conv1d
MAMBA_FORCE_BUILD=TRUE uv pip install --no-deps --no-cache-dir --no-build-isolation mamba-ssm

echo ">>> Installing dwconv"
uv pip install dwconv

echo ">>> Verifying the install"
uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run --no-sync python -c "import torch_cluster; print('torch_cluster', torch_cluster.__version__)"
uv run --no-sync python -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"
uv run --no-sync python -c "import torchsparse; print('torchsparse', torchsparse.__version__)"
uv run --no-sync python -c "import spconv; print('spconv', spconv.__version__)"
uv run --no-sync python -c "import ocnn; print('ocnn', ocnn.__version__)"
uv run --no-sync python -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)"
uv run --no-sync python -c "import torch_pointcloud; print('torch_pointcloud', torch_pointcloud.__version__)"
