#!/usr/bin/env bash
set -euo pipefail

# Install torch-pointcloud for one CUDA / torch combination (builder stage of docker/Dockerfile).
# The wheel sources and pins mirror the selector in docs/installation.md; keep the two in step.
#
# Usage:
#   install.sh CUDA_VERSION TORCH_VERSION [EXTRAS]
#   install.sh 12.6.3 2.8.0 pyg-lib,spconv,ocnn,lightning
#
# EXTRAS is a comma-separated subset of:
#   pyg-lib spconv ocnn dwconv torchsparse mamba flash-attn sptr lightning

CUDA_VERSION="$1"
TORCH_VERSION="$2"
EXTRAS="${3:-}"

IFS=. read -r CUDA_MAJOR CUDA_MINOR _ <<<"${CUDA_VERSION}"
IFS=. read -r _ TORCH_MINOR TORCH_PATCH <<<"${TORCH_VERSION}"
CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
# The Astral GPU index encodes the build in a local version segment.
ASTRAL_LOCAL="+cu.${CUDA_MAJOR}.${CUDA_MINOR}.torch.2.${TORCH_MINOR}"

# `--no-config` skips [tool.uv] in pyproject.toml: `constraint-dependencies` pins the development torch and
# `exclude-newer` hides recently published wheels.
PIP="uv pip install --no-config"

for extra in ${EXTRAS//,/ }; do
    case "${extra}" in
        pyg-lib | spconv | ocnn | dwconv | torchsparse | mamba | flash-attn | sptr | lightning) ;;
        *)
            echo ">>> Unknown extra '${extra}'" >&2
            exit 1
            ;;
    esac
done

has_extra() {
    [[ ",${EXTRAS}," == *",$1,"* ]]
}

# ocnn requires an unpinned torchvision and lightning an unpinned torch: without this constraint a later
# install resolves them from PyPI and replaces the CUDA build.
export UV_CONSTRAINT=/tmp/constraints.txt
echo "torch==${TORCH_VERSION}+${CUDA_TAG}" >"${UV_CONSTRAINT}"

MODULES=(torch_geometric torch_scatter torch_sparse torch_cluster torch_pointcloud)

echo ">>> Installing torch ${TORCH_VERSION} (${CUDA_TAG})"
$PIP "torch==${TORCH_VERSION}+${CUDA_TAG}" --index-url "${TORCH_INDEX}"

echo ">>> Installing torch-pointcloud"
if has_extra lightning; then
    $PIP ".[lightning]" pytest
    MODULES+=(lightning torchmetrics)
else
    $PIP . pytest
fi

# `--only-binary` fails on a missing wheel instead of falling back to a source build from PyPI.
echo ">>> Installing the PyG extensions (FPS, kNN, scatter pooling)"
PYG=(torch-scatter torch-sparse torch-cluster)
if has_extra pyg-lib; then
    PYG+=(pyg-lib)
    MODULES+=(pyg_lib)
fi
$PIP --only-binary :all: "${PYG[@]}" -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

ASTRAL=()
if has_extra flash-attn; then
    ASTRAL+=("flash-attn==2.8.3.post1${ASTRAL_LOCAL}")
    MODULES+=(flash_attn)
fi
if has_extra mamba; then
    # mamba-ssm 2.3.2 requires triton >= 3.5, while torch pins triton 3.4 up to torch 2.8.
    if [ "${TORCH_MINOR}" -ge 9 ]; then
        ASTRAL+=("mamba-ssm==2.3.2.post1${ASTRAL_LOCAL}" "causal-conv1d==1.6.2.post1${ASTRAL_LOCAL}")
    else
        ASTRAL+=("mamba-ssm==2.3.1${ASTRAL_LOCAL}" "causal-conv1d==1.6.1${ASTRAL_LOCAL}")
    fi
    MODULES+=(mamba_ssm causal_conv1d)
fi
if [ "${#ASTRAL[@]}" -gt 0 ]; then
    echo ">>> Installing ${ASTRAL[*]} (prebuilt by the Astral GPU index)"
    $PIP --only-binary :all: "${ASTRAL[@]}" --index "https://wheels.astral.sh/simple/${CUDA_TAG}/"
fi

if has_extra spconv; then
    case "${CUDA_TAG}" in
        cu118 | cu121 | cu124 | cu126) SPCONV="spconv-${CUDA_TAG}" ;;
        # spconv publishes no cu128 build; the cu126 wheel runs on the CUDA 12.8 runtime.
        cu128) SPCONV="spconv-cu126" ;;
        *)
            echo ">>> spconv publishes no build for ${CUDA_TAG}" >&2
            exit 1
            ;;
    esac
    echo ">>> Installing ${SPCONV}"
    $PIP "${SPCONV}"
    MODULES+=(spconv)
fi

if has_extra ocnn; then
    echo ">>> Installing ocnn (OctFormer)"
    # torchvision 0.(15 + minor).patch is the release paired with torch 2.minor.patch.
    $PIP "torchvision==0.$((TORCH_MINOR + 15)).${TORCH_PATCH}+${CUDA_TAG}" --index-url "${TORCH_INDEX}"
    $PIP ocnn
    MODULES+=(ocnn)
fi

if has_extra dwconv || has_extra torchsparse || has_extra sptr; then
    echo ">>> Installing the build backend of the source-built extensions"
    $PIP setuptools wheel
fi

if has_extra dwconv; then
    echo ">>> Installing dwconv (OctFormer, builds from source)"
    $PIP --no-build-isolation \
        "dwconv @ git+https://github.com/octree-nn/dwconv.git@ae53057eaf36dab01aa2727fcc93a749fd995af5"
    MODULES+=(dwconv)
fi

if has_extra torchsparse; then
    echo ">>> Installing torchsparse (SPVCNN, builds from source)"
    # Its setup.py falls back to a CPU-only build when no GPU is visible, which is always the case in `docker build`.
    FORCE_CUDA=1 $PIP --no-deps --no-build-isolation \
        "torchsparse @ git+https://github.com/mit-han-lab/torchsparse.git@385f5ce8718fcae93540511b7f5832f4e71fd835"
    # `--no-deps` above keeps torchsparse from pulling its own torch, so its import-time
    # requirements come in here. `rootpath` imports nothing but the stdlib, yet declares tox,
    # coverage and codecov as runtime deps, so it goes in without them.
    $PIP --no-deps rootpath
    $PIP "backports.cached-property"
    # Not added to MODULES: torchsparse queries the GPU at import, and none is visible during `docker build`.
fi

if has_extra sptr; then
    echo ">>> Installing sptr (SphereFormer, builds from source)"
    # Upstream installs only the CUDA extension, leaving the `sptr` python package (the
    # get_indices_params / sparse_self_attention API) unimportable, and never declares timm.
    # This fork fixes both; tracking https://github.com/JIA-Lab-research/SparseTransformer/pull/10.
    $PIP --no-build-isolation \
        "sptr @ git+https://github.com/arthurdjn/SparseTransformer.git@fix/install-python-package"
    MODULES+=(sptr)
fi

echo ">>> Verifying the install"
python - "${TORCH_VERSION}+${CUDA_TAG}" "${MODULES[@]}" <<'PY'
import importlib
import sys

import torch

expected, *names = sys.argv[1:]
print(f"{'torch':<16} {torch.__version__}  cuda={torch.version.cuda}")
assert torch.__version__ == expected, f"expected torch {expected}"
for name in names:
    module = importlib.import_module(name)
    print(f"{name:<16} {getattr(module, '__version__', 'ok')}")
PY
