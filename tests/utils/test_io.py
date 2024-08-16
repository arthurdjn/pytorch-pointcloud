import json
import textwrap
from typing import Any
from unittest.mock import mock_open, patch

import numpy as np
import pytest
import torch

from torch_pointcloud.utils.io import load_json, load_off


@pytest.mark.parametrize(
    "data",
    [
        {"name": "test", "value": 42},
        {"name": "yet_another_test", "value": {"nested": [1, 2, 3]}},
        [{"name": "test", "value": 42}, {"name": "another_test", "value": 0}],
    ],
)
def test_load_json(data: Any) -> None:
    with patch("builtins.open", mock_open(read_data=json.dumps(data))):
        result = load_json("dummy_path.json")

    assert result == data


@pytest.mark.parametrize(
    "data, expected_nodes, expected_faces",
    [
        (
            """OFF
        3 1 0
        0.0 0.0 0.0
        1.0 0.0 0.0
        0.0 1.0 0.0
        3 0 1 2
        """,
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0, 1, 2]]),
        ),
        (
            """OFF
        8 6 0
        -0.500000 -0.500000 0.500000
        0.500000 -0.500000 0.500000
        -0.500000 0.500000 0.500000
        0.500000 0.500000 0.500000
        -0.500000 0.500000 -0.500000
        0.500000 0.500000 -0.500000
        -0.500000 -0.500000 -0.500000
        0.500000 -0.500000 -0.500000
        4 0 1 3 2
        4 2 3 5 4
        4 4 5 7 6
        4 6 7 1 0
        4 1 7 5 3
        4 6 0 2 4
        """,
            torch.tensor(
                [
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                ]
            ),
            torch.tensor(
                [
                    [0, 1, 3],
                    [0, 3, 2],
                    [2, 3, 5],
                    [2, 5, 4],
                    [4, 5, 7],
                    [4, 7, 6],
                    [6, 7, 1],
                    [6, 1, 0],
                    [1, 7, 5],
                    [1, 5, 3],
                    [6, 0, 2],
                    [6, 2, 4],
                ]
            ),
        ),
    ],
)
def test_load_off(data: str, expected_nodes: np.ndarray, expected_faces: np.ndarray) -> None:
    data = textwrap.dedent(data)
    with patch("builtins.open", mock_open(read_data=data)):
        nodes, faces = load_off("dummy_path.off")

    # Check that the parsed nodes and faces match the expected values
    np.testing.assert_array_equal(nodes, expected_nodes)
    np.testing.assert_array_equal(faces, expected_faces)


def test_load_off_empty_file() -> None:
    with patch("builtins.open", mock_open(read_data="")):
        with pytest.raises(ValueError):
            load_off("dummy_path.off")
