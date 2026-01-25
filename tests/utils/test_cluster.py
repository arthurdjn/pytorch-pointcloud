from unittest.mock import Mock, patch, sentinel

import pytest
import torch

from torch_pointcloud.utils.cluster import fps
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE


def test_fps_with_ratio_and_num_nodes() -> None:
    """Test that the utility fps raises a ValueError if both ratio and num_nodes are provided."""
    with pytest.raises(ValueError, match="Only one of `ratio` or `num_nodes` can be provided."):
        fps(sentinel.src, ratio=sentinel.ratio, num_nodes=sentinel.num_nodes)


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
@patch("torch_pointcloud.utils.cluster.torch_cluster.fps")
def test_fps_with_ratio(mock_fps: Mock) -> None:
    """Test that the utility fps wraps the torch_cluster.fps function and passes the correct arguments."""
    out = fps(
        sentinel.src,
        batch=sentinel.batch,
        ratio=sentinel.ratio,
        batch_size=sentinel.batch_size,
        ptr=sentinel.ptr,
        random_start=sentinel.random_start,
    )

    mock_fps.assert_called_once_with(
        sentinel.src,
        batch=sentinel.batch,
        ratio=sentinel.ratio,
        random_start=sentinel.random_start,
        batch_size=sentinel.batch_size,
        ptr=sentinel.ptr,
    )
    assert out is mock_fps.return_value


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
def test_fps_with_num_nodes_repeat() -> None:
    """Test fps with num_nodes using real tensors.
    Verifies that if num_nodes > number of points, the first selected node is repeated."""
    src = torch.cat(
        [
            torch.tensor([[0.0], [1.0], [10.0], [3.0], [2.0], [4.0]]),
            torch.tensor([[0.0], [4.0], [1.0]]),
        ]
    )
    batch = torch.cat([torch.zeros(6), torch.ones(3)]).long()

    out = fps(src, batch=batch, num_nodes=5, random_start=False)

    assert out.shape == (10,)
    assert torch.equal(out[:5], torch.tensor([0, 2, 5, 4, 1]))
    assert torch.equal(out[5:], torch.tensor([6, 7, 8, 6, 6]))
