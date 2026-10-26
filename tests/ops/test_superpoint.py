import torch

from torch_pointcloud.ops.superpoint import superpoint_instance_masks, superpoint_pool


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
