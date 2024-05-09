import os
from distutils.sysconfig import get_config_vars

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(flag for flag in opt.split() if flag != "-Wstrict-prototypes")

src = "torch_pointcloud/csrc"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

setup(
    name="torch_pointcloud",
    version="1.0",
    install_requires=["torch", "numpy"],
    # packages=["torch_test"],
    # packages=find_packages(),
    # package_dir={"torch_test": "torch_test"},
    packages=["torch_pointcloud"],
    # package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="torch_pointcloud._C",
            sources=sources,
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
            define_macros=[("WITH_CUDA", None)],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
