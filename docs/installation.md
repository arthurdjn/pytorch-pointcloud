# Installation

## Package

```bash
pip install torch-pointcloud
```

This installs the library together with :pytorch: [`torch`](https://pytorch.org) and :pyg: [`torch-geometric`](https://pytorch-geometric.readthedocs.io/).

## PyTorch and CUDA extensions

Select a package manager, torch version, compute platform, and the extras you need; the command below
updates accordingly. Options without a matching wheel are grayed out. `flash-attn`, `mamba`, and the
:pyg: [PyG extensions](https://data.pyg.org/whl/) install as prebuilt wheels, from the
:astral: [Astral GPU indexes](https://wheels.astral.sh/) and the PyG wheel index.

| Extra         | Required by                                                           |
| ------------- | --------------------------------------------------------------------- |
| `pyg-lib`     | FPS, kNN, and scatter pooling (`torch-scatter`, `torch-cluster`, ...) |
| `flash-attn`  | Point Transformer V3, Sonata, Concerto, Utonia                        |
| `mamba`       | Point-Mamba, Voxel-Mamba, LION                                        |
| `spconv`      | SpUNet, SPFormer-UNet, voxel-based detectors                          |
| `ocnn`        | OctFormer (installed together with `dwconv`)                          |
| `torchsparse` | SPVCNN                                                                |
| `sptr`        | SphereFormer                                                          |
| `lightning`   | The Lightning training modules                                        |

<div class="install-selector" id="install-selector" markdown="0">
  <div class="isel-row"><span class="isel-label">Package manager</span><span class="isel-opts">
    <button data-dim="pm" data-val="uv" class="isel-active">uv</button>
    <button data-dim="pm" data-val="pip">pip</button>
    <button data-dim="pm" data-val="conda">conda</button>
  </span></div>
  <div class="isel-row"><span class="isel-label">PyTorch</span><span class="isel-opts">
    <button data-dim="torch" data-val="2.9">2.9</button>
    <button data-dim="torch" data-val="2.10" class="isel-active">2.10</button>
    <button data-dim="torch" data-val="2.11">2.11</button>
    <button data-dim="torch" data-val="2.12">2.12</button>
    <button data-dim="torch" data-val="2.13">2.13</button>
  </span></div>
  <div class="isel-row"><span class="isel-label">Compute</span><span class="isel-opts">
    <button data-dim="cuda" data-val="cpu">CPU</button>
    <button data-dim="cuda" data-val="cu126">CUDA 12.6</button>
    <button data-dim="cuda" data-val="cu128" class="isel-active">CUDA 12.8</button>
    <button data-dim="cuda" data-val="cu130">CUDA 13.0</button>
    <button data-dim="cuda" data-val="cu132">CUDA 13.2</button>
  </span></div>
  <div class="isel-row"><span class="isel-label">Extras</span><span class="isel-opts">
    <button data-dim="extra" data-val="pyg" class="isel-active">pyg-lib</button>
    <button data-dim="extra" data-val="flash">flash-attn</button>
    <button data-dim="extra" data-val="mamba">mamba</button>
    <button data-dim="extra" data-val="spconv">spconv</button>
    <button data-dim="extra" data-val="ocnn">ocnn</button>
    <button data-dim="extra" data-val="torchsparse">torchsparse</button>
    <button data-dim="extra" data-val="sptr">sptr</button>
    <button data-dim="extra" data-val="lightning">lightning</button>
  </span></div>
  <div class="isel-output">
    <button class="isel-copy" id="isel-copy" title="Copy to clipboard" aria-label="Copy to clipboard"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M19 21H8V7h11m0-2H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2m-3-4H4a2 2 0 0 0-2 2v14h2V3h12V1Z"/></svg></button>
    <pre><code id="isel-command"></code></pre>
  </div>
</div>

The combination tested in CI and used for the benchmark results is `torch==2.10.0` with CUDA 12.8.
Other torch or CUDA versions and exact wheel pins are listed on the :astral: [Astral GPU indexes](https://wheels.astral.sh/)
and the :pyg: [PyG wheel index](https://data.pyg.org/whl/).

## Development

To work on the library, clone it and set up the environment with :uv: [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/arthurdjn/pytorch-pointcloud.git
cd pytorch-pointcloud
uv sync
```

For the test and docs tooling, install the dev group and all extras:

```bash
uv sync --all-extras --dev
```

Similarly to the above section, install the extras dependencies for your machine (CUDA 12.8, torch 2.10.0, etc.).

We provide several commands to help you get started:

```bash
make test   # Run the test suite
make format # Format the code
make lint   # Lint the code
make type   # Type check the code
make clean  # Clean the build artifacts
make docs   # Build the documentation
make serve  # Serve the documentation locally
```

## Compatibility

- **Python**: 3.10+
- **PyTorch**: the library requires `torch>=2.5`. The tested combination is `torch==2.10.0` with CUDA 12.8
  wheels; the selector above covers `2.9` to `2.13` across CPU and CUDA 12.6 to 13.2.
- **CUDA**: optional for the point-based families (PointNet, PointNet++, DGCNN, PointNeXt, PointMLP, PointConv, PointCNN, RandLA-Net, and similar), which run inference and training on CPU. The sparse-voxel and flash-attention families (Point Transformer V3, Sonata, Concerto, Utonia, SpUNet, SPVCNN, OctFormer, and the voxel-based detectors) require a CUDA device and their optional dependencies (`spconv`, `torchsparse`, `ocnn`, `flash-attn`).
