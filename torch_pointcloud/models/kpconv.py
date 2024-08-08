import math
import warnings
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter

from torch_pointcloud.utils.config import CACHE_DIR


def kpconv_kernel(
    radius: int,
    num_kernel_points: int,
    dimension: int,
    fixed: Literal["vertical", "horizontal", "none"] = "vertical",
    use_lloyd: bool = False,
) -> None:
    # Load cached kernel if exists
    kernel_path = Path(CACHE_DIR, "kernels", f"k_{num_kernel_points}_{fixed}_{dimension}D.ply")
    if not kernel_path.exists():
        pass

    # Too many points
    if num_kernel_points > 30:
        warnings.warn("Too many points for kernel point optimization. You shoukd use Lloyds instead (lloyd=True).")

    # Check if already done
    if not exists(kernel_file):
        if use_lloyd:
            # Create kernels
            kernel_points = spherical_Lloyd(1.0, num_kpoints, dimension=dimension, fixed=fixed, verbose=0)

        else:
            # Create kernels
            kernel_points, grad_norms = kernel_point_optimization_debug(
                1.0, num_kpoints, num_kernels=100, dimension=dimension, fixed=fixed, verbose=0
            )

            # Find best candidate
            best_k = np.argmin(grad_norms[-1, :])

            # Save points
            kernel_points = kernel_points[best_k, :, :]

        write_ply(kernel_file, kernel_points, ["x", "y", "z"])

    else:
        data = read_ply(kernel_file)
        kernel_points = np.vstack((data["x"], data["y"], data["z"])).T

    # Random roations for the kernel
    # N.B. 4D random rotations not supported yet
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != "vertical":
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

    elif dimension == 3:
        if fixed != "vertical":
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        else:
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

            R = R.astype(np.float32)

    # Add a small noise
    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)

    # Scale kernels
    kernel_points = radius * kernel_points

    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)

    return kernel_points.astype(np.float32)
