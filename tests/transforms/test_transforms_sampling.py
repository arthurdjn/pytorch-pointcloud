from typing import Any, Dict
from unittest.mock import sentinel

import pytest
import torch

import torch_pointcloud.transforms as T
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE


def test_random_sample_preserves_correspondence() -> None:
    pos = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    normal = torch.arange(20, 30, dtype=torch.float32).reshape(10, 1)
    other = torch.tensor([42.0])
    data = {"pos": pos, "normal": normal, "other": other}

    gen = torch.Generator().manual_seed(0)
    result = T.RandomSample(keys=["pos", "normal"], num_samples=5, generator=gen)(data)

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
    gen = torch.Generator().manual_seed(0)
    result = T.RandomSample(keys=["pos"], num_samples=10, replace=True, generator=gen)(data)
    assert result["pos"].shape == (10, 2)


def test_random_sample_determinism() -> None:
    data = {"pos": torch.randn(50, 3)}
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = T.RandomSample(keys=["pos"], num_samples=10, generator=g1)(data)
    b = T.RandomSample(keys=["pos"], num_samples=10, generator=g2)(data)
    assert torch.equal(a["pos"], b["pos"])


def test_random_sample_face_vertices_basic() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    face = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data = {"vertices": vertices, "face": face, "other": sentinel.other}

    gen = torch.Generator().manual_seed(0)
    transform = T.RandomSampleFaceVertices(
        keys=["vertices"], face_key="face", normal_key="normal", num_samples=5, generator=gen
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
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    a = T.RandomSampleFaceVertices(keys=["vertices"], face_key="face", num_samples=4, generator=g1)(
        {"vertices": vertices, "face": face}
    )
    b = T.RandomSampleFaceVertices(keys=["vertices"], face_key="face", num_samples=4, generator=g2)(
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
    g = torch.Generator().manual_seed(0)
    out = T.RandomDropout(keys=("pos", "color"), p_drop=0.5, generator=g)({"pos": pos.clone(), "color": color.clone()})
    assert out["pos"].shape[0] == out["color"].shape[0]
    # Surviving (pos, color) pairs match the original mapping.
    for i in range(out["pos"].shape[0]):
        src_idx = int(out["pos"][i, 0].item()) // 2
        assert out["color"][i].item() == src_idx


def test_random_dropout_invalid_p_drop() -> None:
    with pytest.raises(ValueError, match=r"p_drop"):
        T.RandomDropout(keys="pos", p_drop=1.0)


def test_shuffle_point_preserves_correspondence_and_count() -> None:
    pos = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    color = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    g = torch.Generator().manual_seed(0)
    out = T.ShufflePoint(keys=("pos", "color"), generator=g)({"pos": pos.clone(), "color": color.clone()})
    assert out["pos"].shape == pos.shape
    # Per-row correspondence is preserved.
    for i in range(10):
        src_idx = int(out["pos"][i, 0].item()) // 2
        assert out["color"][i].item() == src_idx


def test_shuffle_point_determinism() -> None:
    pos = torch.randn(20, 3)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = T.ShufflePoint(keys="pos", generator=g1)({"pos": pos.clone()})
    b = T.ShufflePoint(keys="pos", generator=g2)({"pos": pos.clone()})
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
    pytest.param(T.RandomDropout(keys=["pos", "color"], p_drop=0.5, dst_index_key="index"), id="RandomDropout"),
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
    pytest.param(T.RandomDropout(keys=["pos", "color"], p_drop=0.5), id="RandomDropout"),
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
    g = torch.Generator().manual_seed(0)
    out = T.Compose(
        [
            T.RandomSample(keys=["pos", "color"], num_samples=8, generator=g, dst_index_key="index"),
            T.Slice(keys=["pos", "color"], stop=4, dst_index_key="index"),
        ],
    )(scene)
    assert out["index"].shape == (4,)
    assert torch.equal(scene["pos"][out["index"]], out["pos"])
    out = T.Compose(
        [
            T.ApplyMask(keys=["pos", "color"], mask_key="mask", dst_index_key="index"),
            T.RandomSample(keys=["pos", "color"], num_samples=3, generator=g, dst_index_key="index"),
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
        T.RandomDropout(keys=["pos"], p_drop=0.5, p=0.0, dst_index_key="index"),
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
