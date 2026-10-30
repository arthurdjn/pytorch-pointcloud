import pytest
import torch
from torch import Tensor

from torch_pointcloud.ops.superpoint import point_mask_index, superpoint_instance_masks, superpoint_pool

_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def _brute_force_pool(x: Tensor, superpoint: Tensor, batch: Tensor) -> tuple[Tensor, list[int], list[int]]:
    pairs = list(zip(batch.tolist(), superpoint.tolist()))
    keys = sorted(set(pairs))
    index = [keys.index(pair) for pair in pairs]
    features = torch.stack([x[[i == key for i in index]].mean(dim=0) for key in range(len(keys))])
    return features, index, [scene for scene, _ in keys]


def _brute_force_instance_masks(
    instance: list[int], label: list[int], superpoint: list[int], superpoint_batch: list[int]
) -> tuple[list[list[bool]], list[int], list[int]]:
    points: dict[tuple[int, int], list[int]] = {}
    for point, (id_, class_, index) in enumerate(zip(instance, label, superpoint)):
        if id_ >= 0 and class_ != -1:
            points.setdefault((superpoint_batch[index], id_), []).append(point)
    keys = sorted(points)
    sizes = [superpoint.count(index) for index in range(len(superpoint_batch))]
    masks = [
        [
            sum(superpoint[point] == index for point in points[key]) / max(size, 1) > 0.5
            for index, size in enumerate(sizes)
        ]
        for key in keys
    ]
    return masks, [label[points[key][0]] for key in keys], [scene for scene, _ in keys]


POOL_CASES = [
    pytest.param([7, 7, 0, 0, 7], [0, 0, 0, 1, 1], id="ids-repeat-across-scenes"),
    pytest.param([2**62, -(2**62), 5, -1, 2**62, 5], [0, 0, 0, 1, 1, 1], id="negative-and-large-ids"),
    pytest.param([4, 1, 4, 1, 9, 4, 1], [1, 0, 0, 1, 1, 0, 2], id="unsorted-batch"),
    pytest.param([5], [0], id="one-point"),
    pytest.param([0, 0, 0, 0, 0, 0], [3, 1, 4, 0, 5, 2], id="one-superpoint-per-scene"),
]


@pytest.mark.parametrize(("superpoint", "batch"), POOL_CASES)
def test_superpoint_pool_equals_brute_force(superpoint: list[int], batch: list[int]) -> None:
    """Superpoints are the sorted (scene, id) pairs, for any integer ids and any point order."""
    x = torch.randn(len(superpoint), 3, generator=torch.Generator().manual_seed(0))
    features, index, superpoint_batch = superpoint_pool(x, torch.tensor(superpoint), torch.tensor(batch))
    expected = _brute_force_pool(x, torch.tensor(superpoint), torch.tensor(batch))
    torch.testing.assert_close(features, expected[0])
    assert (index.tolist(), superpoint_batch.tolist()) == expected[1:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(("superpoint", "batch"), POOL_CASES)
def test_superpoint_pool_equals_brute_force_cuda(superpoint: list[int], batch: list[int]) -> None:
    x = torch.randn(len(superpoint), 3, generator=torch.Generator().manual_seed(0))
    features, index, superpoint_batch = superpoint_pool(
        x.cuda(),
        torch.tensor(superpoint).cuda(),
        torch.tensor(batch).cuda(),
    )
    expected = _brute_force_pool(x, torch.tensor(superpoint), torch.tensor(batch))
    torch.testing.assert_close(features.cpu(), expected[0])
    assert (index.tolist(), superpoint_batch.tolist()) == expected[1:]


INSTANCE_MASK_CASES = [
    pytest.param(
        [10**9, 10**9, 3, 3, -1, 10**9], [2, 2, 1, 1, -1, 2], [0, 0, 1, 2, 2, 3], [0, 0, 1, 1], id="large-ids"
    ),
    pytest.param(
        [3, 3, 3, 3, 8, 8], [1, 1, 1, 1, 0, 0], [0, 1, 2, 3, 0, 3], [0, 1, 0, 1], id="ids-repeat-across-scenes"
    ),
    pytest.param([5, -1, 5, 5, 2, 2], [4, 0, 4, 4, 1, -1], [2, 0, 2, 1, 0, 1], [1, 0, 1], id="unsorted-scenes"),
]


@pytest.mark.parametrize(("instance", "label", "superpoint", "superpoint_batch"), INSTANCE_MASK_CASES)
def test_superpoint_instance_masks_equals_brute_force(
    instance: list[int], label: list[int], superpoint: list[int], superpoint_batch: list[int]
) -> None:
    """Instances are the sorted (scene, id) pairs of the valid points, for any integer ids and any point order."""
    inputs = [torch.tensor(values) for values in (instance, label, superpoint, superpoint_batch)]
    masks, labels, batch = superpoint_instance_masks(*inputs)
    expected = _brute_force_instance_masks(instance, label, superpoint, superpoint_batch)
    assert (masks.tolist(), labels.tolist(), batch.tolist()) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(("instance", "label", "superpoint", "superpoint_batch"), INSTANCE_MASK_CASES)
def test_superpoint_instance_masks_equals_brute_force_cuda(
    instance: list[int], label: list[int], superpoint: list[int], superpoint_batch: list[int]
) -> None:
    inputs = [torch.tensor(values).cuda() for values in (instance, label, superpoint, superpoint_batch)]
    masks, labels, batch = superpoint_instance_masks(*inputs)
    expected = _brute_force_instance_masks(instance, label, superpoint, superpoint_batch)
    assert (masks.tolist(), labels.tolist(), batch.tolist()) == expected


def test_superpoint_pool_keys_superpoints_by_scene() -> None:
    """Two scenes reuse the ids 0 and 7: they stay separate superpoints, numbered scene by scene."""
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [2.0, 2.0], [4.0, 6.0]])
    superpoint = torch.tensor([7, 7, 0, 0, 7])
    batch = torch.tensor([0, 0, 0, 1, 1])
    features, index, superpoint_batch = superpoint_pool(x, superpoint, batch)
    assert features.tolist() == [[5.0, 4.0], [2.0, 1.0], [2.0, 2.0], [4.0, 6.0]]
    assert index.tolist() == [1, 1, 0, 2, 3]
    assert superpoint_batch.tolist() == [0, 0, 1, 1]


def test_superpoint_pool_matches_per_scene_pooling() -> None:
    torch.manual_seed(0)
    lengths = [50, 80]
    x = torch.randn(sum(lengths), 4)
    superpoint = torch.cat([torch.randint(0, 6, (n,)) * 3 for n in lengths])
    batch = torch.repeat_interleave(torch.arange(2), torch.tensor(lengths))
    features, index, superpoint_batch = superpoint_pool(x, superpoint, batch)
    torch.testing.assert_close(
        features[index[:50]], torch.stack([x[:50][superpoint[:50] == s].mean(0) for s in superpoint[:50]])
    )
    assert torch.equal(superpoint_batch[index], batch)
    assert (superpoint_batch.diff() >= 0).all()


def test_superpoint_instance_masks_majority_threshold() -> None:
    """A superpoint joins an instance when more than `threshold` of all its points, ignored ones included, do."""
    instance = torch.tensor([0, 0, 0, 1, -1, 5, 5, 5])
    label = torch.tensor([2, 2, 2, -1, 2, 4, 4, 4])
    superpoint = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3])
    superpoint_batch = torch.tensor([0, 0, 1, 1])
    masks, labels, batch = superpoint_instance_masks(instance, label, superpoint, superpoint_batch)
    assert masks.tolist() == [[True, False, False, False], [False, False, True, True]]
    assert labels.tolist() == [2, 4]
    assert batch.tolist() == [0, 1]

    masks, _, _ = superpoint_instance_masks(instance, label, superpoint, superpoint_batch, threshold=0.3)
    assert masks[0].tolist() == [True, True, False, False]


def test_superpoint_instance_masks_without_instances() -> None:
    masks, labels, batch = superpoint_instance_masks(
        torch.tensor([-1, -1]), torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([0, 0])
    )
    assert masks.shape == (0, 2) and labels.numel() == 0 and batch.numel() == 0


MASK_INDEX_CASES = [
    pytest.param([[0, 0, 1], [0, 2, 1]], [2, 0, 1, 0, 2], id="unsorted-superpoints"),
    pytest.param([[0, 1, 1, 2], [1, 0, 1, 1]], [1, 1, 0, 1], id="overlapping-masks"),
    pytest.param([[0, 0], [1, 3]], [0, 1, 1, 0], id="superpoint-without-points"),
    pytest.param([[], []], [0, 1, 2], id="no-pairs"),
    pytest.param([[0, 1], [0, 1]], [], id="no-points"),
]


@pytest.mark.parametrize(("mask_index", "superpoint"), MASK_INDEX_CASES)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_REQUIRES_CUDA)])
def test_point_mask_index_equals_the_dense_expansion(
    mask_index: list[list[int]],
    superpoint: list[int],
    device: str,
) -> None:
    """Pair by pair, the points of each covered superpoint, in ascending order."""
    expected = [
        (instance, point)
        for instance, column in zip(*mask_index)
        for point, index in enumerate(superpoint)
        if index == column
    ]
    out = point_mask_index(
        torch.tensor(mask_index, dtype=torch.long, device=device),
        torch.tensor(superpoint, dtype=torch.long, device=device),
    )
    assert out.shape == (2, len(expected))
    assert out.t().tolist() == [list(pair) for pair in expected]
