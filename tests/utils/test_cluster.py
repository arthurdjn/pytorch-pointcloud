from unittest.mock import Mock, patch, sentinel

import pytest
import torch

from torch_pointcloud.utils.cluster import fps, group
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


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
def test_group_shapes_and_overloads() -> None:
    """group densifies a variable-length packed batch into a regular $(B, G, k, 3)$ tensor; the
    return_indices overload adds the flat neighbor index without changing center / neighborhood."""
    torch.manual_seed(0)
    # two scenes of DIFFERENT sizes in one packed batch
    pos = torch.cat([torch.randn(1024, 3), torch.randn(2048, 3)])
    batch = torch.cat([torch.zeros(1024), torch.ones(2048)]).long()

    neighborhood, center = group(pos, batch, num_group=64, group_size=32)
    assert neighborhood.shape == (2, 64, 32, 3)
    assert center.shape == (2, 64, 3)

    neighborhood_idx, center_idx, idx = group(pos, batch, num_group=64, group_size=32, return_indices=True)
    assert idx.shape == (2 * 64 * 32,)
    assert torch.equal(center, center_idx)
    assert torch.equal(neighborhood, neighborhood_idx)


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
def test_group_recenters_on_its_centers() -> None:
    """Each neighborhood is recentered on its center, which is its own nearest neighbor, so every
    group contains an exactly-zero relative position."""
    torch.manual_seed(0)
    pos = torch.randn(512, 3)
    batch = torch.zeros(512, dtype=torch.long)

    neighborhood, _ = group(pos, batch, num_group=32, group_size=16)
    has_self = (neighborhood.abs().sum(dim=-1) == 0).any(dim=-1)
    assert bool(has_self.all())


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
def test_group_idx_indexes_packed_input() -> None:
    """The returned idx is the flat neighbor index into the packed input: gathering pos at idx and
    recentering reproduces the neighborhood exactly (two scenes of different sizes)."""
    torch.manual_seed(0)
    pos = torch.cat([torch.randn(800, 3), torch.randn(900, 3)])
    batch = torch.cat([torch.zeros(800), torch.ones(900)]).long()

    neighborhood, center, idx = group(pos, batch, num_group=40, group_size=16, return_indices=True)
    gathered = pos[idx].view(2, 40, 16, 3) - center.unsqueeze(2)
    assert torch.equal(gathered, neighborhood)


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
def test_group_scene_smaller_than_num_group() -> None:
    """A scene with fewer points than num_group still densifies (FPS repeats points), so the packed
    $(B, G, k, 3)$ shape holds even for degenerate scenes mixed with normal ones."""
    torch.manual_seed(0)
    pos = torch.cat([torch.randn(40, 3), torch.randn(1024, 3)])
    batch = torch.cat([torch.zeros(40), torch.ones(1024)]).long()

    neighborhood, center = group(pos, batch, num_group=64, group_size=8)
    assert neighborhood.shape == (2, 64, 8, 3)
    assert center.shape == (2, 64, 3)


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
def test_group_deterministic_without_random_start() -> None:
    """random_start=False makes the grouping reproducible run-to-run."""
    torch.manual_seed(0)
    pos = torch.randn(1024, 3)
    batch = torch.zeros(1024, dtype=torch.long)

    first = group(pos, batch, num_group=64, group_size=32, random_start=False)
    second = group(pos, batch, num_group=64, group_size=32, random_start=False)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@patch("torch_pointcloud.utils.cluster.knn")
@patch("torch_pointcloud.utils.cluster.fps")
def test_group_calls_fps_and_knn_with_correct_params(mock_fps: Mock, mock_knn: Mock) -> None:
    """group delegates to the internal fps / knn: fps gets the cloud, the batch, num_nodes=num_group
    and the threaded random_start; knn gets the cloud, the FPS centers, group_size and the matching
    per-sample batch indices. fps / knn are mocked, so no torch_cluster is needed."""
    pos = torch.randn(8, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    idx_center = torch.tensor([0, 1, 4, 5])
    row = torch.arange(4).repeat_interleave(3)
    col = torch.tensor([0, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 7])
    mock_fps.return_value = idx_center
    mock_knn.return_value = (row, col)

    neighborhood, center, idx = group(pos, batch, num_group=2, group_size=3, random_start=True, return_indices=True)

    mock_fps.assert_called_once()
    fps_args, fps_kwargs = mock_fps.call_args
    assert fps_args[0] is pos
    assert fps_args[1] is batch
    assert fps_kwargs["num_nodes"] == 2
    assert fps_kwargs["random_start"] is True

    mock_knn.assert_called_once()
    knn_args, knn_kwargs = mock_knn.call_args
    assert knn_args[0] is pos
    assert torch.equal(knn_args[1], pos[idx_center])
    assert knn_args[2] == 3
    assert knn_kwargs["batch_x"] is batch
    assert torch.equal(knn_kwargs["batch_y"], batch[idx_center])

    assert neighborhood.shape == (2, 2, 3, 3)
    assert center.shape == (2, 2, 3)
    assert idx.shape == (12,)
