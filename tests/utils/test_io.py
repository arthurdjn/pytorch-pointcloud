from pathlib import Path

import numpy as np
import pytest
import torch

from torch_pointcloud.utils.io import load_json, load_off, load_safetensors, save_safetensors

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
