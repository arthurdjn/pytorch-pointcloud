ARG CUDA_VERSION=12.6.3
ARG CUDNN_TAG=cudnn
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_TAG}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.8.0
ARG EXTRAS=pyg-lib,spconv,ocnn,lightning
# No GPU is visible during `docker build`, so source-built extensions (torchsparse, mamba,
# flash-attn) cannot query the device: target a fixed set of architectures instead.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git build-essential ninja-build libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN uv venv --python "${PYTHON_VERSION}" "${VIRTUAL_ENV}"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src src
COPY docker/install.sh docker/install.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    bash docker/install.sh "${CUDA_VERSION}" "${TORCH_VERSION}" "${EXTRAS}"

FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_TAG}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=builder /opt/uv /opt/uv
COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace
CMD ["python"]
