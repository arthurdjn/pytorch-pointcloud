import json
from typing import Any, Dict, Tuple

import numpy as np

from .types import PATH_LIKE


def load_json(file_path: PATH_LIKE) -> Dict[str, Any]:
    with open(file_path) as f:
        return json.load(f)


def load_off(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_name, "r") as f:
        lines = f.read().splitlines()

    # skip header
    if lines[0] == "OFF":
        lines = lines[1:]

    # get metadata
    num_nodes, num_faces, *_ = map(int, lines[0].split())

    # load nodes
    nodes_txt = "\n".join(lines[1 : 1 + num_nodes])
    nodes = np.fromstring(nodes_txt, sep=" ").reshape(num_nodes, -1)

    # load faces
    faces_txt = "\n".join(lines[1 + num_nodes : 1 + num_nodes + num_faces])
    faces_idxs = np.fromstring(faces_txt, sep=" ").reshape(num_faces, -1)
    triangles = faces_idxs[faces_idxs[:, 0] == 3, 1:]
    rectangles = faces_idxs[faces_idxs[:, 0] == 4, 1:]

    if rectangles.size > 0:
        first, second = rectangles[:, [0, 1, 2]], rectangles[:, [0, 2, 3]]
        triangles = np.concatenate([triangles, first, second], axis=0)

    return nodes, triangles
