import pytest
import torch

import torch_pointcloud.transforms as T
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE, _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_basic() -> None:
    # Two points that fall in the same voxel + one in another voxel
    pos = torch.tensor([[0.05, 0.05, 0.05], [0.06, 0.06, 0.06], [1.0, 1.0, 1.0]])
    data = {"pos": pos, "feat": torch.tensor([[1.0], [3.0], [5.0]])}
    result = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        keys=["feat"],
        reduce=["mean"],
    )(data)
    assert result["pos"].shape[0] <= 3
    assert result["feat"].shape == (result["pos"].shape[0], 1)


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_with_dst_inverse_key() -> None:
    pos = torch.tensor([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        dst_inverse_key="inverse",
    )({"pos": pos})
    # `inverse` maps each original point to its voxel index
    assert result["inverse"].shape == (3,)
    assert result["inverse"][0] == result["inverse"][1]
    assert result["inverse"][0] != result["inverse"][2]


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_grid_pos_key() -> None:
    pos = torch.tensor([[0.05, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        dst_pos_grid_key="pos_grid",
    )({"pos": pos})
    assert "pos_grid" in result
    assert result["pos_grid"].dtype == torch.long
    assert result["pos_grid"].shape == result["pos"].shape


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_grid_pos_reduce() -> None:
    pos = torch.tensor([[0.05, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = T.Voxelize(
        pos_key="pos",
        pos_reduce="grid",
        size=0.1,
    )({"pos": pos})
    assert result["pos"].dtype == torch.long


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_default_reduce_works_and_keeps_integer_dtype() -> None:
    """`reduce=None` (the default) averages float keys and picks a representative for integer keys."""
    pos = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.9, 1.9, 1.9]])
    segment = torch.tensor([4, 4, 7])
    color = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.25, 0.25, 0.25]])
    result = T.Voxelize(pos_key="pos", pos_reduce="mean", size=0.5, keys=["segment", "color"])(
        {"pos": pos, "segment": segment, "color": color}
    )
    assert result["segment"].dtype == torch.long
    assert sorted(result["segment"].tolist()) == [4, 7]
    assert result["color"].dtype == torch.float32
    row = result["segment"].tolist().index(4)
    assert torch.allclose(result["color"][row], torch.tensor([0.5, 0.5, 0.5]))


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_integer_key_non_first_reduce_stays_integer() -> None:
    pos = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.9, 1.9, 1.9]])
    segment = torch.tensor([4, 6, 7])
    result = T.Voxelize(pos_key="pos", pos_reduce="mean", size=0.5, keys=["segment"], reduce="max")(
        {"pos": pos, "segment": segment}
    )
    assert result["segment"].dtype == torch.long
    assert sorted(result["segment"].tolist()) == [6, 7]


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_integer_min_max_exact_above_float32_precision() -> None:
    """Integer min/max reduce natively; a float32 detour would collapse values above $2^{24}$."""
    pos = torch.zeros(2, 3)
    result = T.Voxelize(pos_key="pos", pos_reduce="mean", size=1.0, keys=["segment"], reduce="min")(
        {"pos": pos, "segment": torch.tensor([2**24 + 1, 2**24 + 2])}
    )
    assert result["segment"].dtype == torch.int64
    assert result["segment"].tolist() == [2**24 + 1]
    result = T.Voxelize(pos_key="pos", pos_reduce="mean", size=1.0, keys=["segment"], reduce="max")(
        {"pos": pos, "segment": torch.tensor([2**24, 2**24 + 1])}
    )
    assert result["segment"].tolist() == [2**24 + 1]


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_first_reduce_picks_first_occurrence_in_input_order() -> None:
    """`first` is the stable first point per voxel, even when voxel members are interleaved."""
    pos = torch.tensor([[0.1, 0.0, 0.0], [1.2, 0.0, 0.0], [0.3, 0.0, 0.0], [1.4, 0.0, 0.0]])
    segment = torch.tensor([10, 20, 30, 40])
    result = T.Voxelize(pos_key="pos", pos_reduce="first", size=1.0, keys=["segment"], reduce="first")(
        {"pos": pos, "segment": segment}
    )
    assert torch.equal(result["pos"], pos[[0, 1]])
    assert torch.equal(result["segment"], torch.tensor([10, 20]))


def test_voxelize_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="size"):
        T.Voxelize(pos_key="pos", pos_reduce="mean", size=0.0)
    with pytest.raises(ValueError, match="pos_reduce"):
        T.Voxelize(pos_key="pos", pos_reduce="median", size=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        T.Voxelize(pos_key="pos", pos_reduce="mean", size=0.5, method="hash")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reduce"):
        T.Voxelize(pos_key="pos", pos_reduce="mean", size=0.5, keys=["segment"], reduce="median")  # type: ignore[arg-type]


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_empty_passthrough(empty_scene: dict) -> None:
    out = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        dst_inverse_key="inverse",
    )(empty_scene)
    assert out["pos"].shape[0] == 0
    assert out["inverse"].shape == (0,)


def test_divisible_pad_default_does_not_write_inverse_key() -> None:
    pos = torch.randn(5, 3)
    batch = torch.zeros(5, dtype=torch.long)
    out = T.DivisiblePad(num_samples=4)({"pos": pos, "batch": batch})
    assert out["pos"].shape[0] == 8  # padded to multiple of 4
    assert "inverse" not in out


def test_divisible_pad_writes_source_to_padded_inverse() -> None:
    """`dst_inverse_key` stores a $(N_\\text{src},) \\to [0, N_\\text{padded})$ map.

    Gathering the padded `pos` with the stored map must recover the original `pos`
    exactly: this is the contract relied on by sliding-window inference.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    batch = torch.zeros(5, dtype=torch.long)
    out = T.DivisiblePad(num_samples=4, dst_inverse_key="inverse")({"pos": pos.clone(), "batch": batch})
    inverse = out["inverse"]
    assert inverse.dtype == torch.long
    assert inverse.shape == (5,)
    assert int(inverse.min()) >= 0 and int(inverse.max()) < out["pos"].shape[0]
    # Round-trip: gather the padded positions back to the source rows
    assert torch.equal(out["pos"][inverse], pos)


def test_divisible_pad_composes_through_prior_inverse() -> None:
    """When `dst_inverse_key` already exists in the dict, the new map composes via gather.

    The composed map is the outer-source -> current-predictor index map: applying it to
    padded positions recovers the outer-source positions one-shot, without intermediate
    bookkeeping.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    # Simulate a prior transform that already wrote an outer-source -> input-row map.
    prior = torch.tensor([0, 1, 2], dtype=torch.long)
    out = T.DivisiblePad(num_samples=4, dst_inverse_key="inverse")(
        {"pos": pos.clone(), "batch": batch, "inverse": prior}
    )
    inverse = out["inverse"]
    assert inverse.shape == (3,)  # length = outer source size, not pre-pad size
    # Composed map gathers from padded back to outer source.
    assert torch.equal(out["pos"][inverse], pos)


def test_divisible_pad_scalar_label_passthrough() -> None:
    """0-dim tensors (e.g. classification labels) are left untouched by the gather."""
    pos = torch.randn(5, 3)
    out = T.DivisiblePad(num_samples=4)({"pos": pos, "label": torch.tensor(3)})
    assert out["pos"].shape[0] == 8
    assert out["label"].ndim == 0
    assert int(out["label"]) == 3


def test_divisible_pad_zero_points_passthrough() -> None:
    pos = torch.zeros(0, 3)
    batch = torch.zeros(0, dtype=torch.long)
    out = T.DivisiblePad(num_samples=4, dst_inverse_key="inverse")({"pos": pos, "batch": batch})
    assert out["pos"].shape == (0, 3)
    # Empty input is a no-op; no inverse needs to be recorded.
    assert "inverse" not in out


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_dst_inverse_key_composes_through_prior() -> None:
    """`Voxelize` composes its source-to-voxel map with an existing inverse via gather."""
    pos = torch.tensor([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [1.0, 0.0, 0.0]])
    prior = torch.tensor([0, 1, 2], dtype=torch.long)
    out = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        dst_inverse_key="inverse",
    )({"pos": pos, "inverse": prior})
    inverse = out["inverse"]
    assert inverse.shape == (3,)
    # Gather voxel-mean positions back to per-source rows; first two map to the same voxel.
    recovered = out["pos"][inverse]
    assert torch.allclose(recovered[0], recovered[1])
    assert not torch.allclose(recovered[0], recovered[2])


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_then_divisible_pad_chain_yields_single_combined_inverse() -> None:
    """Composing Voxelize -> DivisiblePad via a shared `dst_inverse_key` collapses to one map.

    The composed inverse maps each original source row directly to a padded predictor row,
    so a one-shot gather recovers per-source predictions without any intermediate state.
    Voxelize runs pre-collate (no `batch` key), DivisiblePad synthesizes a zero batch.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pipeline = T.Compose(
        [
            T.Voxelize(
                pos_key="pos",
                pos_reduce="mean",
                size=0.1,
                dst_inverse_key="inverse",
            ),
            T.DivisiblePad(num_samples=4, dst_inverse_key="inverse"),
        ]
    )
    out = pipeline({"pos": pos.clone()})
    inverse = out["inverse"]
    assert inverse.shape == (5,)  # length = outer-source size
    n_padded = out["pos"].shape[0]
    assert int(inverse.min()) >= 0 and int(inverse.max()) < n_padded
    # Per-source gather: same-voxel sources land on the same padded row; different voxels split.
    rows = inverse.tolist()
    assert rows[0] == rows[1]  # both in voxel near origin
    assert rows[2] == rows[3]  # both in voxel near x=1
    assert len({rows[0], rows[2], rows[4]}) == 3  # three distinct voxels


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_hard_voxelize_stacks_points_per_voxel() -> None:
    pos = torch.tensor([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [5.5, 5.5, 0.5]])
    x = torch.tensor([[1.0], [2.0], [3.0]])
    out = T.HardVoxelize(
        pos_key="pos",
        feat_key="x",
        voxel_size=(1.0, 1.0, 1.0),
        point_cloud_range=(0.0, 0.0, 0.0, 8.0, 8.0, 8.0),
        max_num_points=2,
        max_num_voxels=10,
    )({"pos": pos, "x": x})
    assert out["voxel"].shape == (2, 2, 4)
    assert out["voxel_num_points"].tolist() == [2, 1]
    assert out["pos_voxel"].tolist() == [[0, 0, 0], [0, 5, 5]]  # (z, y, x) grid indices
    assert torch.allclose(out["voxel"][0, 0], torch.tensor([0.5, 0.5, 0.5, 1.0]))
    assert torch.allclose(out["voxel"][0, 1], torch.tensor([0.6, 0.6, 0.6, 2.0]))
    assert torch.allclose(out["voxel"][1, 0], torch.tensor([5.5, 5.5, 0.5, 3.0]))
    assert torch.allclose(out["voxel"][1, 1], torch.zeros(4))  # padding past the voxel's point count
    assert torch.equal(out["pos"], pos)


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_default_does_not_write_inverse_key() -> None:
    pos = torch.tensor([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = T.Voxelize(pos_key="pos", pos_reduce="mean", size=0.1)({"pos": pos})
    assert out["pos"].shape[0] == 2
    assert "inverse" not in out


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_dst_pos_grid_key_written_for_every_pos_reduce() -> None:
    pos = torch.tensor([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [1.0, 0.0, 0.0]])
    grid = T.Voxelize(pos_key="pos", pos_reduce="grid", size=0.1, dst_pos_grid_key="pos_grid")({"pos": pos})
    first = T.Voxelize(pos_key="pos", pos_reduce="first", size=0.1, dst_pos_grid_key="pos_grid")({"pos": pos})
    assert torch.equal(grid["pos_grid"], grid["pos"])
    assert torch.equal(first["pos_grid"], grid["pos"])
    assert torch.equal(first["pos"], pos[[0, 2]])
