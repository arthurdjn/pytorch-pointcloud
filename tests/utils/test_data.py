# mypy: disable-error-code="list-item"
import pytest
import torch
from torch.utils.data import Dataset

from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader, collate


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


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, torch.tensor([True, False])),
        (1, torch.tensor([1, 2])),
        (1.5, torch.tensor([1.5, 2.5])),
    ],
)
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


def test_collate_strings_are_atomic_in_batch_bookkeeping() -> None:
    # A string is one scalar per sample, not a sequence of characters.
    samples = [
        {"pos": torch.randn(2, 3), "name": "scene_a"},
        {"pos": torch.randn(3, 3), "name": "scene_b"},
    ]
    out = collate(samples, cat_keys=("name",))
    assert out["name"] == ["scene_a", "scene_b"]
    assert out["batch_name"].tolist() == [0, 1]


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


def test_collate_raises_on_key_missing_from_first_sample() -> None:
    # A key only later samples carry must raise, not be silently dropped.
    samples = [
        {"pos": torch.randn(2, 3)},
        {"pos": torch.randn(2, 3), "extra": torch.randn(2, 1)},
    ]
    with pytest.raises(ValueError, match="'extra'.*sample 0"):
        collate(samples)


def test_collate_raises_on_key_missing_from_later_sample() -> None:
    samples = [
        {"pos": torch.randn(2, 3), "extra": torch.randn(2, 1)},
        {"pos": torch.randn(2, 3)},
    ]
    with pytest.raises(ValueError, match="'extra'.*sample 1"):
        collate(samples)


def _detection_sample(num_points: int, num_boxes: int) -> dict[str, torch.Tensor]:
    return {
        DataKeys.POS: torch.randn(num_points, 3),
        DataKeys.BOX: torch.randn(num_boxes, 8),
        DataKeys.LABEL: torch.randint(0, 10, (num_boxes,)),
    }


def test_collate_cat_keys_emits_scene_index() -> None:
    """`cat_keys` keep ragged per-scene tensors packed but add a `batch_<key>` scene index."""
    samples = [_detection_sample(5, 2), _detection_sample(7, 3)]
    out = collate(samples, cat_keys=(DataKeys.BOX,))
    assert out[DataKeys.POS].shape == (12, 3)
    assert out[DataKeys.BATCH].shape == (12,)
    assert out[DataKeys.BOX].shape == (5, 8)
    batch_box = out[f"batch_{DataKeys.BOX}"]
    assert batch_box.shape == (5,)
    assert batch_box.tolist() == [0, 0, 1, 1, 1]


def test_collate_cat_keys_handles_empty_scene() -> None:
    """A scene with no rows contributes nothing to the packed tensor or its scene index."""
    samples = [_detection_sample(4, 0), _detection_sample(6, 2)]
    out = collate(samples, cat_keys=(DataKeys.BOX,))
    assert out[DataKeys.BOX].shape == (2, 8)
    assert out[f"batch_{DataKeys.BOX}"].tolist() == [1, 1]


def test_collate_without_cat_keys_omits_scene_index() -> None:
    """Without `cat_keys` the packed tensor is still concatenated but no scene index is emitted."""
    samples = [_detection_sample(5, 2), _detection_sample(7, 3)]
    out = collate(samples)
    assert out[DataKeys.BOX].shape == (5, 8)
    assert f"batch_{DataKeys.BOX}" not in out
    # Sanity-check the asserted spelling against the key collate really synthesizes.
    with_index = collate(samples, cat_keys=(DataKeys.BOX,))
    assert with_index[f"batch_{DataKeys.BOX}"].tolist() == [0, 0, 1, 1, 1]


def test_collate_key_in_both_stack_and_cat_keys_raises() -> None:
    """A key listed in both `stack_keys` and `cat_keys` is ambiguous and must raise."""
    samples = [_detection_sample(5, 2), _detection_sample(7, 3)]
    with pytest.raises(ValueError, match="stack_keys.*cat_keys"):
        collate(samples, stack_keys=(DataKeys.BOX,), cat_keys=(DataKeys.BOX,))


def test_collate_default_key_specs_do_not_trip_overlap_check() -> None:
    """Plain `collate(samples)` leaves both specs at None; the overlap check must ignore that."""
    samples = [{"pos": torch.randn(2, 3)}, {"pos": torch.randn(3, 3)}]
    out = collate(samples, stack_keys=None, cat_keys=None)
    assert out["pos"].shape == (5, 3)
    assert out["batch"].tolist() == [0, 0, 1, 1, 1]


def test_collate_stack_keys_adds_leading_batch_dim() -> None:
    """`stack_keys` stack dense per-scene tensors to a new leading batch dim instead of concatenating."""

    def scene(num_points: int) -> dict[str, torch.Tensor]:
        return {
            DataKeys.POS: torch.randn(num_points, 3),
            "center_label": torch.randn(64, 3),
            "box_label_mask": torch.zeros(64),
            "vote_label": torch.randn(num_points, 9),
        }

    out = collate([scene(20), scene(20)], stack_keys=("center_label", "box_label_mask", "vote_label"))
    assert out[DataKeys.POS].shape == (40, 3)
    assert out[DataKeys.BATCH].shape == (40,)
    assert out["center_label"].shape == (2, 64, 3)
    assert out["box_label_mask"].shape == (2, 64)
    assert out["vote_label"].shape == (2, 20, 9)


class DummyDetectionDataset(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def test_dataloader_defaults_to_packed_collate() -> None:
    """With no spec the loader packs per-point tensors and synthesizes the per-point batch index."""
    samples = [_detection_sample(5, 2), _detection_sample(7, 3)]
    loader = PointCloudDataLoader(DummyDetectionDataset(samples), batch_size=2)
    batch = next(iter(loader))
    assert batch[DataKeys.POS].shape == (12, 3)
    assert batch[DataKeys.BATCH].tolist() == [0] * 5 + [1] * 7
    assert f"batch_{DataKeys.BOX}" not in batch


def test_dataloader_batch_index_restarts_per_batch() -> None:
    """Each yielded batch gets its own 0-based scene index, independent of dataset position."""
    samples = [_detection_sample(n, 1) for n in (2, 3, 4, 5)]
    loader = PointCloudDataLoader(DummyDetectionDataset(samples), batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][DataKeys.BATCH].tolist() == [0, 0, 1, 1, 1]
    assert batches[1][DataKeys.BATCH].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_dataloader_stack_keys_stack_to_leading_batch_dim() -> None:
    r"""`stack_keys` stack dense per-scene tensors to $(B, M, \cdot)$ instead of concatenating."""
    samples = [
        {DataKeys.POS: torch.randn(5, 3), "center_label": torch.randn(64, 3)},
        {DataKeys.POS: torch.randn(7, 3), "center_label": torch.randn(64, 3)},
    ]
    loader = PointCloudDataLoader(DummyDetectionDataset(samples), batch_size=2, stack_keys=("center_label",))
    batch = next(iter(loader))
    assert batch[DataKeys.POS].shape == (12, 3)
    assert batch["center_label"].shape == (2, 64, 3)
    assert torch.equal(batch["center_label"][0], samples[0]["center_label"])
    assert torch.equal(batch["center_label"][1], samples[1]["center_label"])


def test_dataloader_cat_keys_synthesize_scene_index() -> None:
    """`cat_keys` keep ragged keys packed and emit the `batch_<key>` scene index."""
    samples = [_detection_sample(5, 2), _detection_sample(7, 3)]
    loader = PointCloudDataLoader(DummyDetectionDataset(samples), batch_size=2, cat_keys=(DataKeys.BOX,))
    batch = next(iter(loader))
    assert batch[DataKeys.BOX].shape == (5, 8)
    assert batch[f"batch_{DataKeys.BOX}"].tolist() == [0, 0, 1, 1, 1]


def test_dataloader_forwards_batch_from_and_batch_key() -> None:
    """`batch_from` picks the key sized against and `batch_key` renames the synthesized index."""
    samples = [{"feat": torch.randn(2, 4)}, {"feat": torch.randn(3, 4)}]
    loader = PointCloudDataLoader(DummyDetectionDataset(samples), batch_size=2, batch_from="feat", batch_key="b")
    batch = next(iter(loader))
    assert "batch" not in batch
    assert batch["b"].tolist() == [0, 0, 1, 1, 1]


def test_dataloader_collate_fn_kwarg_overrides_spec() -> None:
    """An explicit `collate_fn` kwarg replaces the packed-batch collate entirely."""
    samples = [_detection_sample(2, 1), _detection_sample(3, 1)]
    loader = PointCloudDataLoader(DummyDetectionDataset(samples), batch_size=2, collate_fn=list)
    batch = next(iter(loader))
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert batch[0][DataKeys.POS].shape == (2, 3)


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
        (DataKeys.INVERSE, "inverse"),
        (DataKeys.NORM_POS, "norm_pos"),
        (DataKeys.NAME, "name"),
        (DataKeys.SCENE, "scene"),
        (DataKeys.SEQUENCE, "sequence"),
        (DataKeys.ROOM_MAX, "room_max"),
        (DataKeys.SCENE_MAX, "scene_max"),
        (DataKeys.BLOCK_CENTER, "block_center"),
        (DataKeys.POINT_INDICES, "point_indices"),
        (DataKeys.SCENE_INDEX, "scene_index"),
        (DataKeys.NUM_SCENE_POINTS, "num_scene_points"),
        (DataKeys.OCTREE, "octree"),
        (DataKeys.POINTS, "points"),
        (DataKeys.BOX_MASK, "box_mask"),
    ],
)
def test_data_keys_string_values(member: DataKeys, expected: str) -> None:
    """DataKeys is a StrEnum used as dict keys across the codebase; lock its surface."""
    assert member == expected
    assert isinstance(member, str)
