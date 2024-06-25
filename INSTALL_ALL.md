To install all deps:

```bash
conda create -n torch-pointcloud python=3.10

pip install numpy==1.26.4
pip install torch torchvision
pip install tensorboard tqdm pyyaml addict pandas

# install Open3D from source
# https://www.open3d.org/docs/release/compilation.html#install-dependencies
# ALSO after make sure to install gcc to avoid 'GLIBCXX_3.4.30' not found error
# conda install -c conda-forge gcc=12

# Install torch-points-kernels
cd torch-points-kernels
pip install -v -e .

# install torch-pointcloud
cd torch-pointcloud
pip install -v -e .

# install torch-geometric
# https://github.com/pyg-team/pytorch_geometric/tree/c9ec0de2aacc33e9dca04907d751d73c0f7f80ee?tab=readme-ov-file#pytorch-23
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html 

# IF gcc errors
# https://stackoverflow.com/questions/39455741/gcc-error-trying-to-exec-cc1plus-execvp-no-such-file-or-directory
# conda install gcc_linux-64
# conda install gxx_linux-64
```
