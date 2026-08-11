import torch

from torch_pointcloud.utils.heatmap import (
    draw_gaussian_to_heatmap,
    draw_heatmap_targets,
    gaussian_radius,
    transpose_gather,
)


def test_gaussian_radius_monotonic_in_size() -> None:
    sizes = torch.tensor([2.0, 4.0, 8.0, 16.0])
    r = gaussian_radius(sizes, sizes)
    assert torch.all(r[1:] > r[:-1])


def test_gaussian_radius_symmetric_in_args() -> None:
    height = torch.tensor([3.0, 7.0, 11.0])
    width = torch.tensor([5.0, 2.0, 9.0])
    assert torch.allclose(gaussian_radius(height, width), gaussian_radius(width, height))


def test_draw_gaussian_peaks_at_one_at_center() -> None:
    hm = torch.zeros(21, 21)
    draw_gaussian_to_heatmap(hm, torch.tensor([10.0, 10.0]), radius=4)
    assert torch.isclose(hm[10, 10], torch.tensor(1.0))
    assert hm.max().item() == hm[10, 10].item()


def test_draw_gaussian_is_symmetric() -> None:
    hm = torch.zeros(21, 21)
    draw_gaussian_to_heatmap(hm, torch.tensor([10.0, 10.0]), radius=4)
    assert torch.allclose(hm, hm.flip(0))
    assert torch.allclose(hm, hm.flip(1))
    assert torch.allclose(hm, hm.t())


def test_draw_gaussian_max_combines() -> None:
    hm = torch.zeros(21, 21)
    draw_gaussian_to_heatmap(hm, torch.tensor([10.0, 10.0]), radius=4)
    draw_gaussian_to_heatmap(hm, torch.tensor([12.0, 10.0]), radius=4)
    assert torch.isclose(hm[10, 10], torch.tensor(1.0))
    assert torch.isclose(hm[10, 12], torch.tensor(1.0))


def test_draw_gaussian_k_scales_peak() -> None:
    hm = torch.zeros(21, 21)
    draw_gaussian_to_heatmap(hm, torch.tensor([10.0, 10.0]), radius=4, k=0.5)
    assert torch.isclose(hm[10, 10], torch.tensor(0.5))


def test_draw_gaussian_out_of_bounds_center_draws_clipped_tail() -> None:
    hm = torch.zeros(10, 10)
    draw_gaussian_to_heatmap(hm, torch.tensor([-1.0, 2.0]), radius=2)
    # Only the in-bounds tail of the Gaussian appears, identical to the same splat on a wider map.
    ref = torch.zeros(10, 30)
    draw_gaussian_to_heatmap(ref, torch.tensor([19.0, 2.0]), radius=2)
    assert torch.allclose(hm, ref[:, 20:30])
    assert hm[2, 0] > hm[2, 1] > 0
    assert hm[:, 2:].sum() == 0


def test_draw_gaussian_fully_outside_center_draws_nothing() -> None:
    hm = torch.zeros(10, 10)
    draw_gaussian_to_heatmap(hm, torch.tensor([-4.0, 2.0]), radius=2)
    assert hm.sum() == 0


def test_draw_heatmap_targets_shapes_and_peak() -> None:
    boxes = torch.tensor([[0.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.3]])
    labels = torch.tensor([0])
    hm, reg, inds, mask = draw_heatmap_targets(
        boxes,
        labels,
        num_classes=3,
        feature_map_size=(16, 20),
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[-4.0, -5.0, -2.0, 4.0, 5.0, 2.0],
        feature_map_stride=1,
        num_max_objs=64,
    )
    assert hm.shape == (3, 20, 16)
    assert reg.shape == (64, 8)
    assert inds.shape == (64,) and inds.dtype == torch.long
    assert mask.shape == (64,) and mask.dtype == torch.long
    assert int(mask.sum()) == 1
    assert torch.isclose(hm.max(), torch.tensor(1.0))
    peak_yx = (hm[0] == hm[0].max()).nonzero()[0]
    assert int(inds[0]) == int(peak_yx[0]) * 16 + int(peak_yx[1])


def test_draw_heatmap_targets_regression_encoding() -> None:
    boxes = torch.tensor([[0.0, 0.0, -1.25, 4.0, 2.0, 1.5, 0.3]])
    labels = torch.tensor([1])
    _, reg, _, mask = draw_heatmap_targets(
        boxes,
        labels,
        num_classes=2,
        feature_map_size=(16, 16),
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[-4.0, -4.0, -2.0, 4.0, 4.0, 2.0],
        feature_map_stride=1,
    )
    assert reg[0, 2].item() == -1.25
    assert torch.allclose(reg[0, 3:6], boxes[0, 3:6].log())
    assert torch.isclose(reg[0, 6], torch.cos(boxes[0, 6]))
    assert torch.isclose(reg[0, 7], torch.sin(boxes[0, 6]))


def test_draw_heatmap_targets_zero_height_box_is_assigned_finite_targets() -> None:
    boxes = torch.tensor([[0.0, 0.0, -1.0, 4.0, 2.0, 0.0, 0.3]])
    labels = torch.tensor([0])
    _, reg, _, mask = draw_heatmap_targets(
        boxes,
        labels,
        num_classes=1,
        feature_map_size=(16, 16),
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[-4.0, -4.0, -2.0, 4.0, 4.0, 2.0],
        feature_map_stride=1,
    )
    assert int(mask.sum()) == 1
    assert torch.isfinite(reg).all()
    assert torch.isclose(reg[0, 5], torch.tensor(1e-5).log())


def test_transpose_gather_reads_channel_vectors_at_flat_indices() -> None:
    feat = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    ind = torch.tensor([[0, 7], [19, 3]])
    out = transpose_gather(feat, ind)
    assert out.shape == (2, 2, 3)
    # Cell 7 of scene 0 is (y=1, x=2): channel c holds feat[0, c, 1, 2].
    assert torch.equal(out[0, 1], feat[0, :, 1, 2])
    assert torch.equal(out[1, 0], feat[1, :, 3, 4])
    assert torch.equal(out[1, 1], feat[1, :, 0, 3])


def test_transpose_gather_matches_draw_heatmap_targets_indices() -> None:
    """Gathering a map built from `y * W + x` indices recovers the per-object peak-cell values."""
    boxes = torch.tensor([[0.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.3]])
    labels = torch.tensor([0])
    _, _, inds, mask = draw_heatmap_targets(
        boxes,
        labels,
        num_classes=1,
        feature_map_size=(16, 20),
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[-4.0, -5.0, -2.0, 4.0, 5.0, 2.0],
        feature_map_stride=1,
        num_max_objs=4,
    )
    feat = torch.arange(20 * 16, dtype=torch.float32).reshape(1, 1, 20, 16)
    out = transpose_gather(feat, inds.unsqueeze(0))
    assert out.shape == (1, 4, 1)
    assert int(mask[0]) == 1
    assert out[0, 0, 0] == float(inds[0])  # feat holds its own flat index at every cell


def test_draw_heatmap_targets_empty() -> None:
    hm, reg, inds, mask = draw_heatmap_targets(
        torch.zeros(0, 7),
        torch.zeros(0, dtype=torch.long),
        num_classes=2,
        feature_map_size=(8, 8),
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[-2.0, -2.0, -2.0, 2.0, 2.0, 2.0],
        feature_map_stride=1,
    )
    assert hm.shape == (2, 8, 8) and hm.sum() == 0
    assert int(mask.sum()) == 0
