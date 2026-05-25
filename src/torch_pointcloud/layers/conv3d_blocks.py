from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch.nn as nn

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm


class Conv3dBlock(nn.Sequential):
    r"""Stacked $\text{Conv3d} \to \text{Norm} \to \text{Act}$ block, mirroring `torch_geometric.nn.MLP`.

    Builds groups of $\text{Conv3d} \to \text{Norm} \to \text{Act}$ (or
    $\text{Conv3d} \to \text{Act} \to \text{Norm}$ when `act_first=True`). When
    `plain_last=True`, the final layer skips norm and activation. Norms are
    resolved via `create_norm` with `dim=3`, so passing `norm="batch_norm"`
    yields `nn.BatchNorm3d`.

    Shape:
        Input: $(B, C_\text{in}, R, R, R)$
        Output: $(B, C_\text{out}, R, R, R)$

    Args:
        channel_list: Channel counts per layer including both endpoints.
        kernel_size: Kernel size for each Conv3d. Pass a sequence to vary per layer.
        act: Activation between layers.
        act_first: If `True`, place activation before normalization.
        act_kwargs: Extra kwargs for the activation.
        norm: Normalization between layers.
        norm_kwargs: Extra kwargs for the normalization.
        plain_last: If `True`, the final layer skips norm and activation.
        bias: Whether the Conv3d layers use a bias term.
    """

    def __init__(
        self,
        channel_list: Sequence[int],
        kernel_size: Union[int, Sequence[int]] = 3,
        *,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        plain_last: bool = True,
        bias: bool = True,
    ):
        channels = list(channel_list)
        if len(channels) < 2:
            raise ValueError(f"channel_list must have at least 2 entries, got {len(channels)}.")
        n_layers = len(channels) - 1
        kernels = [int(kernel_size)] * n_layers if isinstance(kernel_size, int) else [int(k) for k in kernel_size]
        if len(kernels) != n_layers:
            raise ValueError(
                f"kernel_size sequence length ({len(kernels)}) does not match number of layers ({n_layers})."
            )

        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        layers: List[nn.Module] = []
        for i in range(n_layers):
            in_c, out_c = channels[i], channels[i + 1]
            k = kernels[i]
            layers.append(nn.Conv3d(in_c, out_c, k, stride=1, padding=k // 2, bias=bias))
            if i == n_layers - 1 and plain_last:
                continue
            norm_layer = create_norm(norm, out_c, dim=3, **norm_kwargs)
            act_layer = create_act(act, **act_kwargs)
            if act_first:
                if act_layer is not None:
                    layers.append(act_layer)
                if norm_layer is not None:
                    layers.append(norm_layer)
            else:
                if norm_layer is not None:
                    layers.append(norm_layer)
                if act_layer is not None:
                    layers.append(act_layer)

        super().__init__(*layers)
