# Contributing to PyTorch PointCloud

If you are interested in contributing to `torch-pointcloud`, your contributions will likely fall into one of the
following two categories:

1. You want to implement a new feature:
   - In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or
     need help on the design / implementation of your feature, post about it in an issue.
1. You want to fix a bug:
   - Feel free to send a Pull Request (PR) any time you encounter a bug. Please provide a clear and concise description
     of what the bug was. If you are unsure about if this is a bug at all or how to fix, post about it in an issue.

Once you finish implementing a feature or bug-fix, please send a PR to
https://github.com/arthurdjn/pytorch-pointcloud.

## Developing PyTorch PointCloud

To develop `torch-pointcloud` on your machine, here are some tips:

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your_username>/pytorch-pointcloud
   cd pytorch-pointcloud
   ```

1. Install the package in editable mode together with the development tooling, using [`uv`](https://docs.astral.sh/uv/):

   ```bash
   uv sync --all-extras --dev
   ```

   This mode will symlink the Python files from the current local source tree into the Python install.
   Hence, if you modify a Python file, you do not need to re-install the package again.

1. Follow the [installation instructions](https://pytorch-pointcloud.org/latest/installation/) to install the PyG
   kernels (`pyg-lib`) for your PyTorch and CUDA versions. The other extensions (`spconv`,
   `flash-attn`, `mamba`, `ocnn`, `torchsparse`, `sptr`) are optional and only necessary if you develop a feature that
   uses one of these libraries.

   ```bash
   uv pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   ```

   where `${TORCH}` should be replaced by your PyTorch version (*e.g.*, `2.10.0`), and `${CUDA}` should be replaced by
   your CUDA version (*e.g.*, `cpu`, `cu126`, or `cu128`).

1. Ensure that you have a working installation by running the entire test suite with

   ```bash
   make test
   ```

## Unit Testing

The testing suite is located under `tests/` and mirrors `src/`.
Run the entire test suite with

```bash
make test
```

or test individual files via, *e.g.*, `uv run --no-sync pytest tests/transforms/test_functional.py`.
Tests that need an optional dependency or a GPU are skipped when it is missing.

## Continuous Integration

`torch-pointcloud` uses [GitHub Actions](https://github.com/arthurdjn/pytorch-pointcloud/actions) for continuous
integration.

Every time you send a Pull Request, your commit will be built and checked against the project guidelines:

1. Ensure that your code is formatted, linted and typed correctly. We use [`ruff`](https://docs.astral.sh/ruff/) and
   [`mypy`](https://mypy-lang.org/) (strict mode):

   ```bash
   make format
   make lint
   make isort
   make type
   ```

1. Ensure that the entire test suite passes.
   Please feel encouraged to provide a test with your submitted code.

1. Give your PR a [Conventional Commits](https://www.conventionalcommits.org/) title, *e.g.*,
   `fix: handle empty batches in nms3d`. The allowed types are `feat`, `fix`, `docs`, `chore`, `refactor`, `perf`,
   `test`, `build`, `ci` and `revert`, and the subject starts with a lowercase character.

1. Add your feature / bugfix to the [`CHANGELOG.md`](CHANGELOG.md), under the `Unreleased` section.
   If multiple PRs move towards integrating a single feature, it is advised to group them together into one bullet
   point.

## Building Documentation

To build the documentation:

1. [Build and install](#developing-pytorch-pointcloud) the package from source.
1. Generate the documentation via:

   ```bash
   make docs
   ```

1. Or serve it locally with live reload at `127.0.0.1:8000`:

   ```bash
   make serve
   ```

*This guide is adapted from the [PyG contributing guide](https://github.com/pyg-team/pytorch_geometric/blob/master/.github/CONTRIBUTING.md) (MIT License).*
