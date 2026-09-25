# Installation

```bash
pip install torch-pointcloud
```

This installs the library with :pytorch: [`torch`](https://pytorch.org) and :pyg: [`torch-geometric`](https://pytorch-geometric.readthedocs.io/).
The kernels below ship as prebuilt wheels that must match your torch and CUDA build, so install torch first. The
selector writes the commands.

| Dependency    | Needed by                                             |
| ------------- | ----------------------------------------------------- |
| `pyg-lib`     | FPS, kNN and radius search: nearly every model        |
| `flash-attn`  | Point Transformer V3, Sonata, Concerto, Utonia        |
| `mamba`       | PointMamba, Voxel-Mamba, LION                         |
| `spconv`      | SpUNet, SPFormer-UNet, voxel-based detectors          |
| `ocnn`        | OctFormer                                             |
| `torchsparse` | SPVCNN (built from source, needs `libsparsehash-dev`) |
| `sptr`        | SphereFormer (with `torch-scatter`, up to torch 2.12) |
| `lightning`   | The Lightning training modules                        |

<div class="install-selector" id="install-selector" markdown="0">
  <div class="isel-row"><span class="isel-label">Package manager</span><span class="isel-opts">
    <button data-dim="pm" data-val="uv" class="isel-active">uv</button>
    <button data-dim="pm" data-val="pip">pip</button>
    <button data-dim="pm" data-val="conda">conda</button>
  </span></div>
  <div class="isel-row"><span class="isel-label">PyTorch</span><span class="isel-opts">
    <button data-dim="torch" data-val="2.8">2.8</button>
    <button data-dim="torch" data-val="2.9">2.9</button>
    <button data-dim="torch" data-val="2.10" class="isel-active">2.10</button>
    <button data-dim="torch" data-val="2.11">2.11</button>
    <button data-dim="torch" data-val="2.12">2.12</button>
    <button data-dim="torch" data-val="2.13">2.13</button>
    <button data-dim="torch" data-val="2.14">2.14</button>
  </span></div>
  <div class="isel-row"><span class="isel-label">Compute</span><span class="isel-opts">
    <button data-dim="cuda" data-val="cpu">CPU</button>
    <button data-dim="cuda" data-val="cu126">CUDA 12.6</button>
    <button data-dim="cuda" data-val="cu128" class="isel-active">CUDA 12.8</button>
    <button data-dim="cuda" data-val="cu129">CUDA 12.9</button>
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
    <pre><code id="isel-command">uv pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install torch-pointcloud

uv pip install pyg-lib \
  -f https://data.pyg.org/whl/torch-2.10.0+cu128.html</code></pre>
  </div>
</div>

## Compatibility

- Python 3.10+ and `torch>=2.8`. CI runs the tests on CPU with `torch==2.8.0`; the benchmarks use `torch==2.10.0`
  with CUDA 12.8.
- The point-based families (PointNet, PointNet++, DGCNN, PointNeXt, RandLA-Net, ...) run on CPU. The sparse-voxel and
  flash-attention families (Point Transformer V3, Sonata, Concerto, Utonia, SpUNet, SPVCNN, OctFormer, the voxel-based
  detectors) need a CUDA device.

To work on the library itself, see
[`CONTRIBUTING.md`](https://github.com/arthurdjn/pytorch-pointcloud/blob/main/CONTRIBUTING.md).
