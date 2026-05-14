import pytest
import torch

from torch_pointcloud.utils.data import DataKeys, collate


def test_collate_empty_returns_empty_dict() -> None:
    assert collate([]) == {}


def test_collate_single_sample_preserves_tensors() -> None:
    sample = {"pos": torch.randn(5, 3), "label": torch.tensor(7)}
    out = collate([sample])
    assert torch.equal(out["pos"], sample["pos"])
    # Scalar tensors are stacked into a (1,) tensor.
    assert torch.equal(out["label"], torch.tensor([7]))
    # batch_from='pos' is present, so a batch tensor is created.
    assert torch.equal(out["batch"], torch.zeros(5, dtype=torch.long))


def test_collate_concats_matching_tails() -> None:
    samples = [
        {"pos": torch.randn(3, 3), "color": torch.randn(3, 3)},
        {"pos": torch.randn(5, 3), "color": torch.randn(5, 3)},
    ]
    out = collate(samples)
    assert out["pos"].shape == (8, 3)
    assert out["color"].shape == (8, 3)
    assert torch.equal(out["pos"][:3], samples[0]["pos"])
    assert torch.equal(out["pos"][3:], samples[1]["pos"])


def test_collate_stacks_scalar_tensors() -> None:
    samples = [{"pos": torch.randn(2, 3), "label": torch.tensor(i)} for i in range(3)]
    out = collate(samples)
    assert out["label"].shape == (3,)
    assert torch.equal(out["label"], torch.tensor([0, 1, 2]))


def test_collate_mismatched_tails_falls_back_to_list() -> None:
    # Mismatched feature dim -> can't cat along dim 0 -> returned as list.
    samples = [
        {"pos": torch.randn(3, 3), "extra": torch.randn(2, 4)},
        {"pos": torch.randn(5, 3), "extra": torch.randn(2, 7)},
    ]
    out = collate(samples)
    assert out["pos"].shape == (8, 3)
    assert isinstance(out["extra"], list)
    assert len(out["extra"]) == 2


@pytest.mark.parametrize("value,expected", [(True, torch.tensor([True, False])),
                                              (1, torch.tensor([1, 2])),
                                              (1.5, torch.tensor([1.5, 2.5]))])
def test_collate_scalar_python_values_become_tensor(value: object, expected: torch.Tensor) -> None:
    pair = [True, False] if isinstance(value, bool) else ([1, 2] if isinstance(value, int) else [1.5, 2.5])
    samples = [{"pos": torch.randn(1, 3), "v": pair[0]}, {"pos": torch.randn(1, 3), "v": pair[1]}]
    out = collate(samples)
    assert torch.equal(out["v"], expected)


def test_collate_non_tensor_non_scalar_kept_as_list() -> None:
    samples = [
        {"pos": torch.randn(2, 3), "name": "scene_a"},
        {"pos": torch.randn(3, 3), "name": "scene_b"},
    ]
    out = collate(samples)
    assert out["name"] == ["scene_a", "scene_b"]


def test_collate_creates_batch_tensor_from_pos() -> None:
    samples = [{"pos": torch.randn(n, 3)} for n in [2, 3, 5]]
    out = collate(samples)
    expected = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long)
    assert torch.equal(out["batch"], expected)
    assert out["batch"].dtype == torch.long


def test_collate_no_batch_when_batch_from_missing() -> None:
    samples = [{"label": torch.tensor(i)} for i in range(3)]
    out = collate(samples)
    assert "batch" not in out


def test_collate_respects_custom_batch_from_and_batch_key() -> None:
    samples = [{"feat": torch.randn(n, 4)} for n in [2, 3]]
    out = collate(samples, batch_from="feat", batch_key="b")
    assert "batch" not in out
    assert torch.equal(out["b"], torch.tensor([0, 0, 1, 1, 1], dtype=torch.long))


def test_collate_batch_from_1d_tensor() -> None:
    # A 1D tensor's length is its shape[0]; the batch index should match per-element.
    samples = [{"pos": torch.arange(n).float()} for n in [2, 4]]
    out = collate(samples)
    assert out["pos"].shape == (6,)
    assert torch.equal(out["batch"], torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long))


def test_collate_batch_from_scalar_uses_length_one() -> None:
    # A scalar `pos` has ndim<1, so length is treated as 1 per sample.
    samples = [{"pos": torch.tensor(3.0)} for _ in range(4)]
    out = collate(samples)
    assert torch.equal(out["batch"], torch.arange(4, dtype=torch.long))


def test_collate_uses_first_sample_keys() -> None:
    # Only keys from the first sample are collated (no union behavior).
    samples = [
        {"pos": torch.randn(2, 3), "color": torch.randn(2, 3)},
        {"pos": torch.randn(2, 3), "color": torch.randn(2, 3), "extra": torch.randn(2, 1)},
    ]
    out = collate(samples)
    assert set(out.keys()) == {"pos", "color", "batch"}


@pytest.mark.parametrize(
    "member,expected",
    [
        (DataKeys.X, "x"),
        (DataKeys.POS, "pos"),
        (DataKeys.POS_GRID, "pos_grid"),
        (DataKeys.COLOR, "color"),
        (DataKeys.NORMAL, "normal"),
        (DataKeys.FACE, "face"),
        (DataKeys.SEGMENT, "segment"),
        (DataKeys.SEMANTIC, "semantic"),
        (DataKeys.INSTANCE, "instance"),
        (DataKeys.INTENSITY, "intensity"),
        (DataKeys.CATEGORY, "category"),
        (DataKeys.LABEL, "label"),
        (DataKeys.BATCH, "batch"),
        (DataKeys.CLUSTER, "cluster"),
        (DataKeys.OCTREE, "octree"),
        (DataKeys.POINTS, "points"),
        (DataKeys.INBOX_MASK, "inbox_mask"),
    ],
)
def test_data_keys_string_values(member: DataKeys, expected: str) -> None:
    """DataKeys is a StrEnum used as dict keys across the codebase; lock its surface."""
    assert member == expected
    assert isinstance(member, str)
