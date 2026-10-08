"""Options added for the reference training recipes (Pointcept's augmentation conventions)."""

import math

import pytest
import torch

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F


def _scene() -> dict:
    torch.manual_seed(0)
    pos = torch.rand(200, 3) * torch.tensor([4.0, 3.0, 2.5]) + torch.tensor([1.0, -2.0, 0.5])
    normal = torch.nn.functional.normalize(torch.randn(200, 3), dim=1)
    return {"pos": pos, "normal": normal, "box": torch.tensor([[2.0, -1.0, 1.0, 1.0, 0.5, 0.8, 0.3]])}


def test_random_rotate_about_bbox_center_keeps_the_center_fixed() -> None:
    data = _scene()
    center = (data["pos"].min(0).values + data["pos"].max(0).values) / 2
    out = T.RandomRotate(keys="pos", vector_keys="normal", axis=0, angle_range=(30.0, 30.0), center="bbox", p=1.0)(data)
    rotation = F.rotation_matrix(math.radians(30.0), axis=0)
    assert torch.allclose(out["pos"], (data["pos"] - center) @ rotation.T + center, atol=1e-5)
    # the center itself is a fixed point of the rotation
    assert torch.allclose(((center - center) @ rotation.T + center), center)
    # the normals are directions: rotated about the origin, never shifted
    assert torch.allclose(out["normal"], data["normal"] @ rotation.T, atol=1e-5)
    assert torch.allclose(out["normal"].norm(dim=1), torch.ones(200), atol=1e-5)


def test_random_rotate_centroid_and_explicit_center_and_boxes() -> None:
    data = _scene()
    out = T.RandomRotate(keys="pos", box_key="box", angle_range=(90.0, 90.0), center="centroid", p=1.0)(data)
    assert torch.allclose(out["pos"].mean(0), data["pos"].mean(0), atol=1e-5)
    explicit = T.RandomRotate(keys="pos", box_key="box", angle_range=(90.0, 90.0), center=(2.0, -1.0, 0.0), p=1.0)(data)
    assert torch.allclose(explicit["box"][0, :2], data["box"][0, :2], atol=1e-5)  # the box sits on the center
    assert explicit["box"][0, 6] == pytest.approx(0.3 + math.pi / 2)
    origin = T.RandomRotate(keys="pos", angle_range=(90.0, 90.0), p=1.0)(data)
    assert torch.allclose(origin["pos"], data["pos"] @ F.rotation_matrix(math.pi / 2, axis=2).T, atol=1e-5)


def test_random_rotate_rejects_unknown_center() -> None:
    with pytest.raises(ValueError, match="center"):
        T.RandomRotate(keys="pos", center="middle")


def test_color_auto_contrast_random_blend_and_flat_cloud() -> None:
    torch.manual_seed(0)
    color = torch.rand(50, 3) * 0.5 + 0.2
    fixed = T.RandomColorAutoContrast(keys="color", blend=0.7, p=1.0)({"color": color})["color"]
    assert torch.allclose(fixed, F.color_auto_contrast(color, blend=0.7))
    random_a = T.RandomColorAutoContrast(keys="color", blend=None, p=1.0, seed=1)({"color": color})["color"]
    random_b = T.RandomColorAutoContrast(keys="color", blend=None, p=1.0, seed=2)({"color": color})["color"]
    assert not torch.allclose(random_a, random_b)
    # both are blends of the input and its full stretch
    stretched = F.color_auto_contrast(color, blend=1.0)
    weight = ((random_a - color) / (stretched - color)).flatten()
    assert torch.allclose(weight, weight[0].expand_as(weight), atol=1e-4) and 0.0 <= weight[0] <= 1.0
    flat = torch.full((50, 3), 0.3)
    assert torch.equal(F.color_auto_contrast(flat, blend=1.0), flat)


def test_elastic_distortion_runs_several_passes_under_one_gate() -> None:
    data = _scene()
    two = T.RandomElasticDistortion(keys="pos", granularity=(0.2, 0.8), magnitude=(0.4, 1.6), p=1.0, seed=0)
    out = two(data)
    assert out["pos"].shape == data["pos"].shape and not torch.allclose(out["pos"], data["pos"])
    assert two.passes == ((0.2, 0.4), (0.8, 1.6))
    unchanged = T.RandomElasticDistortion(keys="pos", granularity=(0.2, 0.8), magnitude=(0.4, 1.6), p=0.0)(data)
    assert torch.equal(unchanged["pos"], data["pos"])
    with pytest.raises(ValueError, match="same number"):
        T.RandomElasticDistortion(keys="pos", granularity=(0.2, 0.8), magnitude=0.4)


def test_sphere_crop_max_ratio_keeps_a_fraction_of_the_points() -> None:
    torch.manual_seed(0)
    data = {"pos": torch.rand(1000, 3), "segment": torch.randint(0, 3, (1000,))}
    out = T.SphereCrop(pos_key="pos", radius=math.inf, max_ratio=0.6, keys="segment", center="random_point")(data)
    assert out["pos"].shape[0] == 600 and out["segment"].shape[0] == 600
    both = T.SphereCrop(pos_key="pos", radius=math.inf, max_ratio=0.6, max_nodes=100, center="random_point")(data)
    assert both["pos"].shape[0] == 100
    with pytest.raises(ValueError, match="max_ratio"):
        T.SphereCrop(pos_key="pos", radius=1.0, max_ratio=1.5)
