from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, Mock, patch, sentinel

import pytest
import torch
from torch import Tensor

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE


def test_random_sample_preserves_correspondence() -> None:
    pos = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    normal = torch.arange(20, 30, dtype=torch.float32).reshape(10, 1)
    other = torch.tensor([42.0])
    data = {"pos": pos, "normal": normal, "other": other}

    result = T.RandomSample(keys=["pos", "normal"], num_samples=5, seed=0)(data)

    assert result["pos"].shape == (5, 2)
    assert result["normal"].shape == (5, 1)
    # correspondence: row i of result["pos"] and result["normal"] came from the same input row
    for i in range(5):
        src_row = int(result["pos"][i, 0].item()) // 2
        assert torch.equal(result["normal"][i], normal[src_row])
    # untouched key passed through
    assert result["other"] is other
    # input dict not mutated
    assert set(data.keys()) == {"pos", "normal", "other"}
    assert data["pos"] is pos


def test_random_sample_replace_false_upsamples_oversample() -> None:
    data = {"pos": torch.randn(10, 3), "color": torch.randn(10, 3)}
    result = T.RandomSample(keys=["pos", "color"], num_samples=20)(data)
    assert result["pos"].shape[0] == 20
    assert result["color"].shape[0] == 20


def test_random_sample_replace_true_allows_oversample() -> None:
    data = {"pos": torch.arange(6, dtype=torch.float32).reshape(3, 2)}
    result = T.RandomSample(keys=["pos"], num_samples=10, replace=True, seed=0)(data)
    assert result["pos"].shape == (10, 2)


def test_random_sample_determinism() -> None:
    data = {"pos": torch.randn(50, 3)}
    a = T.RandomSample(keys=["pos"], num_samples=10, seed=7)(data)
    b = T.RandomSample(keys=["pos"], num_samples=10, seed=7)(data)
    assert torch.equal(a["pos"], b["pos"])


def test_random_sample_face_vertices_basic() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    face = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data = {"vertices": vertices, "face": face, "other": sentinel.other}

    transform = T.RandomSampleFaceVertices(
        keys=["vertices"],
        face_key="face",
        dst_normal_key="normal",
        num_samples=5,
        seed=0,
    )
    result = transform(data)

    assert result["vertices"].shape == (5, 3)
    assert result["normal"].shape == (5, 3)
    # Z is 0 since the mesh lies in the XY plane
    assert torch.allclose(result["vertices"][:, 2], torch.zeros(5), atol=1e-5)
    assert result["other"] is sentinel.other


def test_random_sample_face_vertices_determinism() -> None:
    vertices = torch.randn(8, 3)
    face = torch.tensor([[0, 1, 2], [3, 4, 5], [5, 6, 7]], dtype=torch.long)
    a = T.RandomSampleFaceVertices(keys=["vertices"], face_key="face", num_samples=4, seed=1)(
        {"vertices": vertices, "face": face}
    )
    b = T.RandomSampleFaceVertices(keys=["vertices"], face_key="face", num_samples=4, seed=1)(
        {"vertices": vertices, "face": face}
    )
    assert torch.equal(a["vertices"], b["vertices"])


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
def test_farthest_point_sample_num_samples() -> None:
    pos = torch.randn(20, 3)
    labels = torch.arange(20)
    data = {"pos": pos, "label": labels, "other": sentinel.other}

    result = T.FarthestPointSample(pos_key="pos", keys=["label"], num_samples=5)(data)
    assert result["pos"].shape == (5, 3)
    assert result["label"].shape == (5,)
    assert result["other"] is sentinel.other
    # subsampled labels must be a subset of input labels
    assert set(result["label"].tolist()).issubset(set(labels.tolist()))


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
def test_farthest_point_sample_ratio() -> None:
    pos = torch.randn(10, 3)
    result = T.FarthestPointSample(pos_key="pos", ratio=0.5)({"pos": pos})
    assert result["pos"].shape[0] == 5


def test_random_dropout_preserves_correspondence() -> None:
    pos = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    color = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    out = T.RandomDropout(keys=("pos", "color"), drop_ratio_range=(0.5, 0.5), seed=0)(
        {"pos": pos.clone(), "color": color.clone()}
    )
    assert out["pos"].shape[0] == out["color"].shape[0]
    # Surviving (pos, color) pairs match the original mapping.
    for i in range(out["pos"].shape[0]):
        src_idx = int(out["pos"][i, 0].item()) // 2
        assert out["color"][i].item() == src_idx


def test_random_dropout_invalid_drop_ratio_range() -> None:
    with pytest.raises(ValueError, match=r"drop_ratio_range"):
        T.RandomDropout(keys="pos", drop_ratio_range=(1.0, 1.0))
    with pytest.raises(ValueError, match=r"drop_ratio_range"):
        T.RandomDropout(keys="pos", drop_ratio_range=(0.3, 0.2))


def test_shuffle_point_preserves_correspondence_and_count() -> None:
    pos = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    color = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    out = T.ShufflePoint(keys=("pos", "color"), seed=0)({"pos": pos.clone(), "color": color.clone()})
    assert out["pos"].shape == pos.shape
    # Per-row correspondence is preserved.
    for i in range(10):
        src_idx = int(out["pos"][i, 0].item()) // 2
        assert out["color"][i].item() == src_idx


def test_shuffle_point_determinism() -> None:
    pos = torch.randn(20, 3)
    a = T.ShufflePoint(keys="pos", seed=7)({"pos": pos.clone()})
    b = T.ShufflePoint(keys="pos", seed=7)({"pos": pos.clone()})
    assert torch.equal(a["pos"], b["pos"])


def test_slice_rows() -> None:
    data = {"pos": torch.arange(12.0).reshape(4, 3)}
    out = T.Slice(keys="pos", stop=2)(data)
    assert torch.equal(out["pos"], torch.arange(6.0).reshape(2, 3))


def test_slice_column_to_new_key() -> None:
    data = {"pos": torch.arange(12.0).reshape(4, 3)}
    out = T.Slice(keys="pos", start=2, stop=3, dim=1, dst_keys="height")(data)
    assert out["height"].shape == (4, 1)
    assert torch.equal(out["height"][:, 0], torch.tensor([2.0, 5.0, 8.0, 11.0]))
    assert out["pos"].shape == (4, 3)


def test_slice_step() -> None:
    data = {"x": torch.arange(10)}
    out = T.Slice(keys="x", step=2)(data)
    assert out["x"].tolist() == [0, 2, 4, 6, 8]


def mesh_scene(index: int) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(index)
    return {
        "pos": torch.randn(16, 3, generator=g),
        "normal": torch.randn(16, 3, generator=g),
        "face": torch.tensor([[0, 1, 2], [2, 3, 4]]),
        "label": torch.tensor(index, dtype=torch.long),
        "name": f"mesh_{index:04d}",
    }


SELECTION_POS = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 3) * 0.3


SELECTION_SCENE = {
    "pos": SELECTION_POS,
    "color": torch.arange(10)[:, None].repeat(1, 3),
    "mask": SELECTION_POS[:, 0] > 1.0,
}


SELECTION_SAMPLERS = [
    pytest.param(T.RandomSample(keys=["pos", "color"], num_samples=4, dst_index_key="index"), id="RandomSample"),
    pytest.param(
        T.FarthestPointSample(pos_key="pos", keys=["color"], num_samples=4, dst_index_key="index"),
        id="FarthestPointSample",
        marks=pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed"),
    ),
    pytest.param(
        T.SphereCrop(pos_key="pos", keys=["color"], radius=1.0, center=(0.0, 0.0, 0.0), dst_index_key="index"),
        id="SphereCrop",
    ),
    pytest.param(
        T.RemoveNearOrigin(pos_key="pos", keys=["color"], radius=0.5, dst_index_key="index"), id="RemoveNearOrigin"
    ),
    pytest.param(
        T.RandomDropout(keys=["pos", "color"], drop_ratio_range=(0.5, 0.5), dst_index_key="index"),
        id="RandomDropout",
    ),
    pytest.param(T.ShufflePoint(keys=["pos", "color"], dst_index_key="index"), id="ShufflePoint"),
    pytest.param(T.ApplyMask(keys=["pos", "color"], mask_key="mask", dst_index_key="index"), id="ApplyMask"),
    pytest.param(T.Slice(keys=["pos", "color"], stop=4, dst_index_key="index"), id="Slice"),
]


DEFAULT_SELECTION_SAMPLERS = [
    pytest.param(T.RandomSample(keys=["pos", "color"], num_samples=4), id="RandomSample"),
    pytest.param(
        T.FarthestPointSample(pos_key="pos", keys=["color"], num_samples=4),
        id="FarthestPointSample",
        marks=pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed"),
    ),
    pytest.param(T.SphereCrop(pos_key="pos", keys=["color"], radius=1.0, center=(0.0, 0.0, 0.0)), id="SphereCrop"),
    pytest.param(T.RemoveNearOrigin(pos_key="pos", keys=["color"], radius=0.5), id="RemoveNearOrigin"),
    pytest.param(T.RandomDropout(keys=["pos", "color"], drop_ratio_range=(0.5, 0.5)), id="RandomDropout"),
    pytest.param(T.ShufflePoint(keys=["pos", "color"]), id="ShufflePoint"),
    pytest.param(T.ApplyMask(keys=["pos", "color"], mask_key="mask"), id="ApplyMask"),
    pytest.param(T.Slice(keys=["pos", "color"], stop=4), id="Slice"),
]


@pytest.mark.parametrize("transform", SELECTION_SAMPLERS)
def test_selection_sampler_index_round_trips(transform: T.DictTransform) -> None:
    scene = SELECTION_SCENE
    out = transform(scene)
    index = out["index"]
    assert index.dtype == torch.long
    assert index.shape == (out["pos"].shape[0],)
    assert torch.equal(scene["pos"][index], out["pos"])
    assert torch.equal(scene["color"][index], out["color"])


@pytest.mark.parametrize("transform", DEFAULT_SELECTION_SAMPLERS)
def test_selection_sampler_default_writes_no_index(transform: T.DictTransform) -> None:
    out = transform(SELECTION_SCENE)
    assert "index" not in out


def test_index_composes_through_prior() -> None:
    scene = SELECTION_SCENE
    out = T.Compose(
        [
            T.RandomSample(keys=["pos", "color"], num_samples=8, seed=0, dst_index_key="index"),
            T.Slice(keys=["pos", "color"], stop=4, dst_index_key="index"),
        ],
    )(scene)
    assert out["index"].shape == (4,)
    assert torch.equal(scene["pos"][out["index"]], out["pos"])
    out = T.Compose(
        [
            T.ApplyMask(keys=["pos", "color"], mask_key="mask", dst_index_key="index"),
            T.RandomSample(keys=["pos", "color"], num_samples=3, seed=0, dst_index_key="index"),
        ]
    )(scene)
    assert out["index"].shape == (3,)
    assert torch.equal(scene["pos"][out["index"]], out["pos"])
    assert torch.equal(scene["color"][out["index"]], out["color"])


def test_slice_column_writes_no_index() -> None:
    out = T.Slice(keys="pos", start=2, stop=3, dim=1, dst_keys="height", dst_index_key="index")(
        {"pos": torch.randn(5, 3)}
    )
    assert out["height"].shape == (5, 1)
    assert "index" not in out


@pytest.mark.parametrize(
    "transform",
    [
        T.RandomDropout(keys=["pos"], drop_ratio_range=(0.5, 0.5), p=0.0, dst_index_key="index"),
        T.ShufflePoint(keys=["pos"], p=0.0, dst_index_key="index"),
        T.SphereCrop(pos_key="pos", radius=1.0, p=0.0, dst_index_key="index"),
    ],
    ids=lambda t: type(t).__name__,
)
def test_p_skipped_sampler_writes_identity_index(transform: T.DictTransform) -> None:
    pos = torch.randn(5, 3)
    out = transform({"pos": pos})
    assert torch.equal(out["index"], torch.arange(5))
    prior = torch.tensor([9, 8, 7, 6, 5])
    out = transform({"pos": pos, "index": prior})
    assert torch.equal(out["index"], prior)


@pytest.fixture
def sample_mesh() -> Tuple[Tensor, Tensor]:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    face = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
        ]
    )
    return vertices, face


def test_random_sample_default_without_replacement() -> None:
    """Default `replace=False` samples without duplicates."""
    tensor = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    result = F.random_sample(tensor, num_samples=5)
    assert result.shape == (5, 1)
    # Without replacement, all sampled values are unique.
    assert result.unique().numel() == 5


def test_random_sample_return_indices() -> None:
    """random_sample returns both the sampled tensor and indices when return_indices=True."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    sampled, indices = F.random_sample(tensor, num_samples=3, return_indices=True)

    assert sampled.shape == (3, 2)
    assert indices.shape == (3,)
    assert torch.equal(sampled, tensor[indices])


def test_random_sample_oversample_without_replace_upsamples() -> None:
    tensor = torch.tensor([[1.0], [2.0]])
    result = F.random_sample(tensor, num_samples=10)
    assert result.shape == (10, 1)


def test_random_sample_oversample_with_replace_ok() -> None:
    tensor = torch.tensor([[1.0], [2.0]])
    result = F.random_sample(tensor, num_samples=10, replace=True)
    assert result.shape == (10, 1)


def test_random_sample_empty_raises() -> None:
    tensor = torch.empty(0, 3)
    with pytest.raises(ValueError, match="empty tensor"):
        F.random_sample(tensor, num_samples=4)


def test_random_sample_empty_zero_samples_ok() -> None:
    tensor = torch.empty(0, 3)
    result = F.random_sample(tensor, num_samples=0)
    assert result.shape == (0, 3)


def test_random_sample_seed_reproducibility() -> None:
    """Test that random_sample produces identical results with the same seed."""
    tensor = torch.randn(100, 3)
    generator = torch.Generator()

    generator.manual_seed(42)
    a = F.random_sample(tensor, num_samples=20, generator=generator)
    generator.manual_seed(42)
    b = F.random_sample(tensor, num_samples=20, generator=generator)
    assert torch.equal(a, b)


def test_random_sample_face_vertices(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape."""
    vertices, face = sample_mesh
    num_samples = 10

    sampled = F.random_sample_face_vertices(vertices, face, num_samples)
    assert sampled.shape == (num_samples, 3)


def test_random_sample_face_vertices_with_normals(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape with normal."""
    vertices, face = sample_mesh
    num_samples = 10

    sampled, normal = F.random_sample_face_vertices(vertices, face, num_samples, return_normals=True)
    assert sampled.shape == (num_samples, 3)
    assert normal.shape == (num_samples, 3)
    assert torch.allclose(torch.norm(normal, dim=1), torch.ones(num_samples))


def test_random_sample_face_vertices_seed_reproducibility(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that random_sample_face_vertices produces identical results with the same seed."""
    vertices, face = sample_mesh
    generator = torch.Generator()

    generator.manual_seed(42)
    a = F.random_sample_face_vertices(vertices, face, num_samples=10, generator=generator)
    generator.manual_seed(42)
    b = F.random_sample_face_vertices(vertices, face, num_samples=10, generator=generator)
    assert torch.equal(a, b)


def test_random_sample_face_vertices_single_face() -> None:
    """Test random_sample_face_vertices with a single-face mesh."""
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    face = torch.tensor([[0, 1, 2]])
    sampled = F.random_sample_face_vertices(vertices, face, num_samples=5)
    assert sampled.shape == (5, 3)


@patch("torch_pointcloud.transforms.sampling.fps")
def test_farthest_point_sample_with_num_samples(mock_fps: Mock) -> None:
    """Test that farthest_point_sample delegates to fps with num_samples."""
    pos = MagicMock()
    num_samples = 10

    result = F.farthest_point_sample(pos, num_samples=num_samples)

    mock_fps.assert_called_once_with(pos, num_nodes=num_samples, ratio=None, random_start=False)
    assert result is mock_fps.return_value


@patch("torch_pointcloud.transforms.sampling.fps")
def test_farthest_point_sample_with_ratio(mock_fps: Mock) -> None:
    """Test that farthest_point_sample delegates to fps with ratio."""
    pos = MagicMock()
    ratio = 0.5

    result = F.farthest_point_sample(pos, ratio=ratio)

    mock_fps.assert_called_once_with(pos, num_nodes=None, ratio=ratio, random_start=False)
    assert result is mock_fps.return_value


@patch("torch_pointcloud.transforms.sampling.fps")
def test_farthest_point_sample_random_start(mock_fps: Mock) -> None:
    """Test that farthest_point_sample delegates to fps with random_start."""
    pos = MagicMock()

    result = F.farthest_point_sample(pos, num_samples=5, random_start=True)

    mock_fps.assert_called_once_with(pos, num_nodes=5, ratio=None, random_start=True)
    assert result is mock_fps.return_value


def test_random_dropout_mask_keep_rate() -> None:
    g = torch.Generator().manual_seed(0)
    mask = F.random_dropout_mask(10000, drop_ratio=0.3, generator=g)
    rate = mask.float().mean().item()
    assert abs(rate - 0.7) < 0.05  # within statistical noise


def test_random_dropout_mask_invalid_drop_ratio() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        F.random_dropout_mask(10, drop_ratio=1.0)


def test_shuffle_indices_is_permutation() -> None:
    g = torch.Generator().manual_seed(0)
    perm = F.shuffle_indices(20, generator=g)
    assert perm.dtype == torch.long
    assert sorted(perm.tolist()) == list(range(20))
