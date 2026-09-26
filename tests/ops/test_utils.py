import torch

from torch_pointcloud.ops.utils import decimate_indices, first_permutation, offset_index


def test_decimate_indices_consecutive_batch_ids() -> None:
    batch = torch.tensor([0, 0, 1, 1, 1, 1])
    indices, decim_batch = decimate_indices(batch, 2, generator=torch.Generator().manual_seed(0))
    assert indices.shape == decim_batch.shape
    assert decim_batch.tolist() == [0, 1, 1]
    assert torch.equal(decim_batch, batch[indices])


def test_decimate_indices_non_consecutive_batch_ids() -> None:
    """Batch ids with gaps (e.g. after filtering a scene) must stay aligned with the returned indices."""
    batch = torch.tensor([0, 0, 0, 0, 2, 2])
    indices, decim_batch = decimate_indices(batch, 2, generator=torch.Generator().manual_seed(0))
    assert indices.shape == decim_batch.shape
    assert decim_batch.tolist() == [0, 0, 2]
    assert torch.equal(decim_batch, batch[indices])


def test_first_permutation_picks_first_occurrence() -> None:
    cluster = torch.tensor([1, 0, 1, 2, 0])
    perm = first_permutation(cluster)
    assert perm.tolist() == [1, 0, 3]


def test_first_permutation_with_explicit_num_clusters() -> None:
    cluster = torch.tensor([0, 0, 1, 1, 1, 2])
    perm = first_permutation(cluster, num_clusters=3)
    assert perm.tolist() == [0, 2, 5]


def test_first_permutation_empty() -> None:
    cluster = torch.empty(0, dtype=torch.long)
    perm = first_permutation(cluster)
    assert perm.shape == (0,)
    assert perm.dtype == torch.long


def test_offset_index_shifts_by_scene_rows() -> None:
    inverse = torch.tensor([0, 1, 1, 0, 2, 2, 1])
    batch_inverse = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    batch = torch.tensor([0, 0, 1, 1, 1])
    assert torch.equal(offset_index(inverse, batch_inverse, batch), torch.tensor([0, 1, 1, 2, 4, 4, 3]))
