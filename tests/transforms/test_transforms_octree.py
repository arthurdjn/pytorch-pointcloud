import pytest
import torch

import torch_pointcloud.transforms as T


def test_build_octree_happy_path() -> None:
    ocnn = pytest.importorskip("ocnn")
    pos = torch.rand(64, 3) * 2 - 1  # cube [-1, 1]
    normal = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    transform = T.BuildOctree(
        pos_key="pos",
        octree_key="octree",
        normal_key="normal",
        depth=4,
    )
    out = transform({"pos": pos, "normal": normal})
    assert "octree" in out
    assert isinstance(out["octree"], ocnn.octree.Octree)


def test_build_octree_with_points_key() -> None:
    ocnn = pytest.importorskip("ocnn")
    pos = torch.rand(32, 3) * 2 - 1
    transform = T.BuildOctree(
        pos_key="pos",
        octree_key="octree",
        points_key="points",
        depth=3,
    )
    out = transform({"pos": pos})
    assert "octree" in out and "points" in out
    assert isinstance(out["octree"], ocnn.octree.Octree)


def test_build_octree_rejects_same_octree_and_points_key() -> None:
    pytest.importorskip("ocnn")
    with pytest.raises(ValueError, match="must be different"):
        T.BuildOctree(pos_key="pos", octree_key="octree", points_key="octree", depth=3)


def test_octree_features_nd() -> None:
    pytest.importorskip("ocnn")
    pos = torch.rand(64, 3) * 2 - 1
    normal = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    data = {"pos": pos, "normal": normal}
    data = T.BuildOctree(
        pos_key="pos",
        octree_key="octree",
        normal_key="normal",
        depth=4,
    )(data)
    out = T.OctreeFeatures(keys=["octree"], features_type="ND", dst_keys=["feat"])(data)
    # "N" = normal (3 channels), "D" = displacement (1 channel) → 4 channels total.
    # ocnn's get_input_feature returns (C, K) where C is channels and K is num nodes.
    assert out["feat"].ndim == 2
    assert 4 in out["feat"].shape
