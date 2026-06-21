#!/usr/bin/env bash
set -euo pipefail

# Optional system dep for torchsparse (Ubuntu/Debian):
# sudo apt-get install -y libsparsehash-dev

echo ">>> Creating a fresh virtual environment"
uv venv --clear

echo ">>> Installing the package + all and dev extras"
uv sync --all-extras --dev

echo ">>> Installing torch + torchvision (CUDA 12.8)"
uv pip install --no-config torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

echo ">>> Installing pyg-lib and friends (CUDA wheels)"
uv pip install --no-config pyg_lib torch-geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# spconv has no cu128 build; the cu126 wheel runs against the CUDA 12.8 runtime.
echo ">>> Installing spconv"
uv pip install --no-config spconv-cu126

echo ">>> Installing ocnn"
uv pip install --no-config ocnn

echo ">>> Installing mamba"
uv pip install --no-config setuptools wheel ninja einops packaging rootpath transformers
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --no-config --no-deps --no-cache-dir --no-build-isolation causal-conv1d
MAMBA_FORCE_BUILD=TRUE uv pip install --no-config --no-deps --no-cache-dir --no-build-isolation mamba-ssm

echo ">>> Installing flash-attn (prebuilt wheel)"
# Official from-source build (slow; needs nvcc + the real torch). Use if the prebuilt wheel above is unavailable:
# uv pip install --no-config --no-deps --no-cache-dir --no-build-isolation flash-attn
uv pip install --no-config --no-deps --no-cache-dir \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp310-cp310-linux_x86_64.whl"

echo ">>> Installing dwconv"
uv pip install --no-config --no-cache-dir --no-build-isolation "dwconv @ git+https://github.com/octree-nn/dwconv.git"

echo ">>> Installing torchsparse"
uv pip install --no-config --no-deps --no-cache-dir --no-build-isolation "torchsparse @ git+https://github.com/mit-han-lab/torchsparse.git@385f5ce8718fcae93540511b7f5832f4e71fd835"

echo ">>> Installing fvdb-core (CUDA 12.8 build from the official fVDB index)"
uv pip install --no-config --no-deps "fvdb-core==0.4.2+pt210.cu128" --extra-index-url https://d36m13axqqhiit.cloudfront.net/simple

echo ">>> Verifying the install"
uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run --no-sync python -c "import torch_cluster; print('torch_cluster', torch_cluster.__version__)"
uv run --no-sync python -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"
uv run --no-sync python -c "import torchsparse; print('torchsparse', torchsparse.__version__)"
uv run --no-sync python -c "import spconv; print('spconv', spconv.__version__)"
uv run --no-sync python -c "import ocnn; print('ocnn', ocnn.__version__)"
uv run --no-sync python -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)"
uv run --no-sync python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
uv run --no-sync python -c "import dwconv; print('dwconv', dwconv is not None)"
uv run --no-sync python -c "import fvdb; print('fvdb', fvdb.__version__)"
uv run --no-sync python -c "import torch_pointcloud; print('torch_pointcloud', torch_pointcloud.__version__)"
