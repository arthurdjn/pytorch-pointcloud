"""Point features of IGN LiDAR HD aerial clouds, as Myria3D builds and normalizes them."""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_tuple_size
from torch_pointcloud.utils.types import KeyCollection

from .base import DictTransform

__all__ = [
    "LidarHDFeatures",
    "StandardizeLidarHDFeatures",
]

LIDAR_HD_MAX_RETURNS = 7.0
LIDAR_HD_MAX_COLOR = 255.0 * 256.0


def lidar_hd_features(
    intensity: Tensor,
    return_number: Tensor,
    number_of_returns: Tensor,
    color: Optional[Tensor] = None,
    nir: Optional[Tensor] = None,
) -> Tensor:
    r"""The 9 point features of a LiDAR HD cloud, in Myria3D's order.

    | Channel | Feature                                                        |
    | ------- | -------------------------------------------------------------- |
    | 0       | intensity, raw                                                 |
    | 1, 2    | return number and number of returns, divided by 7              |
    | 3, 4, 5 | red, green, blue, divided by 65280 ($255 \cdot 256$)           |
    | 6       | near-infrared, divided by 65280                                |
    | 7       | mean colour $(r + g + b) / 3$                                  |
    | 8       | NDVI $(\text{nir} - r) / (\text{nir} + r + 10^{-6})$           |

    The colours come from orthoimages and only describe the first return, so every colour channel is zero where the
    return number is above 1. A missing colour or near-infrared reads as zeros.

    Args:
        intensity: Intensity, shape $(N,)$ or $(N, 1)$.
        return_number: Return number, 1-based, shape $(N,)$ or $(N, 1)$.
        number_of_returns: Number of returns, shape $(N,)$ or $(N, 1)$.
        color: Raw 16-bit RGB, shape $(N, 3)$.
        nir: Raw 16-bit near-infrared, shape $(N,)$ or $(N, 1)$.

    Returns:
        The features, shape $(N, 9)$, `float32`.
    """
    intensity = intensity.reshape(-1, 1).float()
    return_number = return_number.reshape(-1, 1).float()
    number_of_returns = number_of_returns.reshape(-1, 1).float()
    num_points = intensity.size(0)
    color = intensity.new_zeros(num_points, 3) if color is None else color.reshape(-1, 3).float() / LIDAR_HD_MAX_COLOR
    nir = intensity.new_zeros(num_points, 1) if nir is None else nir.reshape(-1, 1).float() / LIDAR_HD_MAX_COLOR

    occluded = return_number > 1
    color = color.masked_fill(occluded, 0.0)
    nir = nir.masked_fill(occluded, 0.0)
    red = color[:, :1]
    rgb_avg = color.mean(dim=1, keepdim=True)
    ndvi = (nir - red) / (nir + red + 1e-6)
    features = [
        intensity,
        return_number / LIDAR_HD_MAX_RETURNS,
        number_of_returns / LIDAR_HD_MAX_RETURNS,
        color,
        nir,
        rgb_avg,
        ndvi,
    ]
    return torch.cat(features, dim=1)


class LidarHDFeatures(DictTransform):
    """Build the 9 LiDAR HD point features into one tensor, see `lidar_hd_features`.

    See Also:
        `torch_pointcloud.transforms.functional.lidar_hd_features`

    Args:
        keys: The keys of the intensity, return number, number of returns, colour and near-infrared, in that order.
        dst_key: The key to store the $(N, 9)$ features in.
        allow_missing_keys: If `True`, a missing colour or near-infrared reads as zeros, as in Myria3D. The other
            three keys are always required.
    """

    def __init__(self, keys: KeyCollection, dst_key: str, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        if len(self.keys) != 5:
            raise ValueError(
                f"Expected the intensity, return number, number of returns, colour and near-infrared keys, "
                f"got {self.keys}."
            )

        self.dst_key = dst_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        present = set(self.iter_keys(data))
        intensity_key, return_number_key, number_of_returns_key, color_key, nir_key = self.keys
        data[self.dst_key] = lidar_hd_features(
            data[intensity_key],
            data[return_number_key],
            data[number_of_returns_key],
            color=data[color_key] if color_key in present else None,
            nir=data[nir_key] if nir_key in present else None,
        )
        return data


def standardize_lidar_hd_features(x: Tensor, clamp_sigma: float = 3.0) -> Tensor:
    r"""Standardize the intensity and mean colour of LiDAR HD features over one cloud, as Myria3D does.

    The intensity (channel 0) is first mapped to $\log(1 + i)$. Each of the two channels is then standardized,
    $c' = (c - \mu) / \sigma$ with $\sigma$ the standard deviation plus $10^{-6}$, and clamped to
    $[-k \sigma, k \sigma]$ with $k$ = `clamp_sigma`. The bound scales with $\sigma$ of the raw channel, not with the
    unit standard deviation of $c'$, which is how the Myria3D checkpoints were trained.

    Args:
        x: Features of one cloud from `lidar_hd_features`, shape $(N, 9)$.
        clamp_sigma: The clamp $k$.

    Returns:
        The standardized features, shape $(N, 9)$.
    """
    x = x.clone()
    for channel, values in ((0, torch.log(x[:, 0] + 1)), (7, x[:, 7])):
        std = torch.nan_to_num(values.std() + 1e-6, nan=1.0)
        bound = clamp_sigma * std
        x[:, channel] = ((values - values.mean()) / std).clamp(min=-bound, max=bound)
    return x


class StandardizeLidarHDFeatures(DictTransform):
    """Standardize the intensity and mean colour of LiDAR HD features, see `standardize_lidar_hd_features`.

    Apply it last, on the points the model sees: the statistics are those of the sampled cloud.

    See Also:
        `torch_pointcloud.transforms.functional.standardize_lidar_hd_features`

    Args:
        keys: The keys of the $(N, 9)$ features built by `LidarHDFeatures`.
        clamp_sigma: The clamp $k$, in standard deviations of the raw channel.
        dst_keys: The keys to store the standardized features in. Defaults to `keys`.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        clamp_sigma: float = 3.0,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.clamp_sigma = clamp_sigma
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = standardize_lidar_hd_features(data[key], clamp_sigma=self.clamp_sigma)
        return data
