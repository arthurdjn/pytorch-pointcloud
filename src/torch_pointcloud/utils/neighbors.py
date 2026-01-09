import torch
from torch import Tensor
from torch_geometric.utils import to_dense_batch


def gaussian_kernel_density(x: Tensor, batch: Tensor, bandwidth: float) -> Tensor:
    r"""Computes the Gaussian Kernel Density (KDE) for a given tensor.

    Args:
        x: The input tensor of shape $(N, C)$.
        batch: The batch tensor of shape $(N,)$.
        bandwidth: The bandwidth of the Gaussian kernel.

    Returns:
        The Gaussian Kernel Density of shape $(N,)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.utils.neighbors import gaussian_kernel_density

        pos = torch.randn(100, 3)
        batch = torch.zeros(100)
        bandwidth = 0.1

        density = gaussian_kernel_density(pos, batch, bandwidth)
        print(density.shape)
        # torch.Size([100])
        ```
    """

    # TODO: Maybe avoid the expansive 'to_dense_batch' operation by using a better suited approach for packed data,
    # TODO: like a custom CUDA kernel to compute directly the KDE?
    # NOTE: This function is approximately 20% slower than the original one (from PointConv paper)
    # https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py
    x_dense, mask = to_dense_batch(x, batch)

    dist = -2 * torch.matmul(x_dense, x_dense.transpose(1, 2))
    sq_norm = torch.sum(x_dense**2, dim=-1)
    dist += sq_norm.unsqueeze(2)
    dist += sq_norm.unsqueeze(1)

    density = torch.exp(-dist / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    mask_neighbors = mask.unsqueeze(1).float()
    density = density * mask_neighbors

    density_sum = density.sum(dim=-1)
    num_nodes = mask.sum(dim=-1).float().clamp(min=1.0)
    x_density = density_sum / num_nodes.unsqueeze(1)
    return x_density[mask].squeeze()
