from typing import Any, Dict, Tuple

import pytest
import torch

from torch_pointcloud.ops.heatmap import (
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


def _draw_heatmap_targets_loop(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The per-object loop the vectorized `draw_heatmap_targets` replaced, kept as the reference."""
    num_classes, feature_map_size = kwargs["num_classes"], kwargs["feature_map_size"]
    voxel_size, point_cloud_range = kwargs["voxel_size"], kwargs["point_cloud_range"]
    feature_map_stride, num_max_objs = kwargs["feature_map_stride"], kwargs["num_max_objs"]
    gaussian_overlap, min_radius = kwargs["gaussian_overlap"], kwargs["min_radius"]
    width, height = feature_map_size
    code_size = boxes.shape[-1] + 1
    heatmap = boxes.new_zeros(num_classes, height, width)
    reg_targets = boxes.new_zeros(num_max_objs, code_size)
    inds = boxes.new_zeros(num_max_objs, dtype=torch.long)
    mask = boxes.new_zeros(num_max_objs, dtype=torch.long)
    if boxes.shape[0] == 0:
        return heatmap, reg_targets, inds, mask

    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    pos_x = torch.clamp((x - point_cloud_range[0]) / voxel_size[0] / feature_map_stride, min=0, max=width - 0.5)
    pos_y = torch.clamp((y - point_cloud_range[1]) / voxel_size[1] / feature_map_stride, min=0, max=height - 0.5)
    center = torch.stack([pos_x, pos_y], dim=-1)
    center_int = center.int()
    center_int_float = center_int.float()
    dx = boxes[:, 3] / voxel_size[0] / feature_map_stride
    dy = boxes[:, 4] / voxel_size[1] / feature_map_stride
    radius = torch.clamp_min(gaussian_radius(dy, dx, min_overlap=gaussian_overlap).int(), min_radius)
    for i in range(min(num_max_objs, boxes.shape[0])):
        if dx[i] <= 0 or dy[i] <= 0:
            continue

        if not (0 <= center_int[i, 0] <= width and 0 <= center_int[i, 1] <= height):
            continue

        draw_gaussian_to_heatmap(heatmap[int(labels[i])], center[i], int(radius[i].item()))
        inds[i] = center_int[i, 1] * width + center_int[i, 0]
        mask[i] = 1
        reg_targets[i, 0:2] = center[i] - center_int_float[i]
        reg_targets[i, 2] = z[i]
        reg_targets[i, 3:6] = boxes[i, 3:6].clamp_min(1e-5).log()
        reg_targets[i, 6] = torch.cos(boxes[i, 6])
        reg_targets[i, 7] = torch.sin(boxes[i, 6])
        if boxes.shape[1] > 7:
            reg_targets[i, 8:] = boxes[i, 7:]
    return heatmap, reg_targets, inds, mask


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_draw_heatmap_targets_matches_per_object_loop(seed: int) -> None:
    """The vectorized drawing equals the per-object reference loop exactly (boxes near and over the map edges,
    big and degenerate boxes, velocity columns, a capped object count)."""
    torch.manual_seed(seed)
    num = 70
    boxes = torch.cat(
        [
            torch.rand(num, 2) * 60 - 30,  # some centers outside the +-24 range: clamped to the border
            torch.rand(num, 1) * 2 - 1,
            torch.rand(num, 3) * 12 + 0.1,  # wide radii
            (torch.rand(num, 1) - 0.5) * 6.28,
            torch.randn(num, 2),  # velocity
        ],
        dim=1,
    )
    boxes[:3, 3] = 0.0  # degenerate boxes are skipped
    labels = torch.randint(0, 4, (num,))
    kwargs: Dict[str, Any] = dict(
        num_classes=4,
        feature_map_size=(48, 40),
        voxel_size=(0.5, 0.6, 1.0),
        point_cloud_range=(-24.0, -24.0, -2.0, 24.0, 24.0, 2.0),
        feature_map_stride=1,
        num_max_objs=64,
        gaussian_overlap=0.1,
        min_radius=2,
    )
    expected = _draw_heatmap_targets_loop(boxes, labels, **kwargs)
    result = draw_heatmap_targets(boxes, labels, **kwargs)
    for got, ref in zip(result, expected):
        assert torch.equal(got, ref)
