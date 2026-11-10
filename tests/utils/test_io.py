from pathlib import Path

import numpy as np
import pytest
import torch

from torch_pointcloud.utils.io import load_json, load_off, load_ply, load_safetensors, save_ply, save_safetensors

OFF_CONTENT = """OFF
# a full-line comment
4 2 0
0.0 0.0 0.0  # an inline comment
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
3 0 1 2
4 0 1 2 3
"""


def test_load_json(tmp_path: Path) -> None:
    file_path = tmp_path / "meta.json"
    file_path.write_text('{"a": 1, "b": [1, 2]}')
    assert load_json(file_path) == {"a": 1, "b": [1, 2]}


def test_load_off_parses_vertices_and_triangulates_quads(tmp_path: Path) -> None:
    file_path = tmp_path / "mesh.off"
    file_path.write_text(OFF_CONTENT)
    nodes, faces = load_off(file_path)
    expected_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    expected_faces = np.array([[0, 1, 2], [0, 1, 2], [0, 2, 3]])  # the quad splits into two triangles
    np.testing.assert_array_equal(nodes, expected_nodes)
    np.testing.assert_array_equal(faces, expected_faces)


def test_load_off_single_line_header(tmp_path: Path) -> None:
    file_path = tmp_path / "mesh.off"
    file_path.write_text("OFF4 2 0\n" + OFF_CONTENT.split("\n", 3)[3])
    nodes, faces = load_off(file_path)
    assert nodes.shape == (4, 3)
    assert faces.shape == (3, 3)


def test_load_off_empty_file_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.off"
    file_path.write_text("OFF\n")
    with pytest.raises(ValueError, match="empty"):
        load_off(file_path)


def test_safetensors_roundtrip(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.safetensors"
    pos = torch.randn(4, 3)
    data = {"pos": pos, "count": 7, "ratio": 0.5, "flag": True, "name": "scene0"}
    save_safetensors(file_path, data)

    loaded = load_safetensors(file_path)
    assert set(loaded.keys()) == set(data.keys())
    assert loaded["name"] == "scene0"
    torch.testing.assert_close(loaded["pos"], pos)
    torch.testing.assert_close(loaded["count"], torch.tensor(7))
    torch.testing.assert_close(loaded["ratio"], torch.tensor(0.5))
    flag = loaded["flag"]
    assert isinstance(flag, torch.Tensor)
    assert torch.equal(flag, torch.tensor(True))


def test_save_safetensors_accepts_non_contiguous_tensors(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.safetensors"
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not x.is_contiguous()
    save_safetensors(file_path, {"x": x})
    loaded = load_safetensors(file_path)
    torch.testing.assert_close(loaded["x"], x)


def test_ply_roundtrip_binary(tmp_path: Path) -> None:
    file_path = tmp_path / "mesh.ply"
    vertex = np.array([(0.0, 1.5, 7), (2.0, -1.0, 255)], dtype=[("x", "f4"), ("y", "f8"), ("label", "u1")])
    face = np.array([([0, 1, 2],), ([2, 1, 0],)], dtype=[("vertex_indices", "i4", (3,))])
    save_ply(file_path, {"vertex": vertex, "face": face})

    assert file_path.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n")
    loaded = load_ply(file_path)
    assert list(loaded) == ["vertex", "face"]
    assert loaded["vertex"].dtype == vertex.dtype
    np.testing.assert_array_equal(loaded["vertex"], vertex)
    assert loaded["face"]["vertex_indices"].shape == (2, 3)
    np.testing.assert_array_equal(loaded["face"]["vertex_indices"], face["vertex_indices"])


def test_ply_roundtrip_ascii(tmp_path: Path) -> None:
    file_path = tmp_path / "mesh.ply"
    vertex = np.array([(0.25, -3, 7), (1e-3, 4, 0)], dtype=[("x", "f4"), ("y", "i2"), ("label", "u1")])
    face = np.array([([0, 1, 2],)], dtype=[("vertex_indices", "i4", (3,))])
    save_ply(file_path, {"vertex": vertex, "face": face}, text=True)

    text = file_path.read_text()
    assert text.startswith("ply\nformat ascii 1.0\n")
    assert text.endswith("end_header\n0.25 -3 7\n0.001 4 0\n3 0 1 2\n")
    loaded = load_ply(file_path)
    np.testing.assert_array_equal(loaded["vertex"], vertex)
    np.testing.assert_array_equal(loaded["face"]["vertex_indices"], face["vertex_indices"])


def test_load_ply_big_endian(tmp_path: Path) -> None:
    file_path = tmp_path / "cloud.ply"
    header = b"ply\nformat binary_big_endian 1.0\ncomment made by hand\nelement vertex 2\nproperty float x\nproperty int label\nend_header\n"
    body = np.array([(1.5, -2), (0.0, 7)], dtype=[("x", ">f4"), ("label", ">i4")]).tobytes()
    file_path.write_bytes(header + body)

    loaded = load_ply(file_path)["vertex"]
    assert loaded.dtype.isnative
    np.testing.assert_array_equal(loaded["x"], np.array([1.5, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(loaded["label"], np.array([-2, 7]))


def test_load_ply_varying_list_lengths(tmp_path: Path) -> None:
    file_path = tmp_path / "polygons.ply"
    file_path.write_text(
        "ply\nformat ascii 1.0\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n3 0 1 2\n4 0 1 2 3\n"
    )
    face = load_ply(file_path)["face"]
    assert face.dtype["vertex_indices"].kind == "O"
    np.testing.assert_array_equal(face["vertex_indices"][0], [0, 1, 2])
    np.testing.assert_array_equal(face["vertex_indices"][1], [0, 1, 2, 3])


def test_load_ply_truncated_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "cloud.ply"
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nend_header\n"
    file_path.write_bytes(header + np.array([1.0], dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="Truncated"):
        load_ply(file_path)


def test_save_ply_rejects_unsupported_dtype(tmp_path: Path) -> None:
    vertex = np.array([(1,)], dtype=[("label", "i8")])
    with pytest.raises(ValueError, match="no type"):
        save_ply(tmp_path / "cloud.ply", {"vertex": vertex})


def test_load_ply_skips_blank_lines_in_ascii(tmp_path: Path) -> None:
    file_path = tmp_path / "cloud.ply"
    file_path.write_bytes(
        b"ply\r\nformat ascii 1.0\r\nelement vertex 2\r\nproperty float x\r\nend_header\r\n1.0\r\n\r\n2.0\r\n"
    )
    np.testing.assert_array_equal(load_ply(file_path)["vertex"]["x"], np.array([1.0, 2.0], dtype=np.float32))


def test_ply_list_next_to_a_property_named_like_its_count(tmp_path: Path) -> None:
    face = np.array([(9, [0, 1, 2]), (8, [2, 1, 0])], dtype=[("v count", "i4"), ("v", "i4", (3,))])
    face.dtype.names = ("v_count", "v")
    file_path = tmp_path / "mesh.ply"
    save_ply(file_path, {"face": face})
    loaded = load_ply(file_path)["face"]
    np.testing.assert_array_equal(loaded["v_count"], [9, 8])
    np.testing.assert_array_equal(loaded["v"], face["v"])


def test_load_ply_without_format_line_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "cloud.ply"
    file_path.write_bytes(b"ply\nelement vertex 1\nproperty float x\nend_header\n1.0\n")
    with pytest.raises(ValueError, match="format"):
        load_ply(file_path)


def test_save_ply_rejects_whitespace_in_names(tmp_path: Path) -> None:
    vertex = np.array([(1.5,)], dtype=[("my field", "f4")])
    with pytest.raises(ValueError, match="whitespace"):
        save_ply(tmp_path / "cloud.ply", {"vertex": vertex})
