import json
import re
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .types import PathLike


def load_json(file_path: PathLike) -> Dict[str, Any]:
    with open(file_path) as f:
        return json.load(f)


def load_off(file_path: PathLike) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(file_path, "r") as f:
        file_content = f.read()

    # Use re to remove comments (both inline and full-line comments)
    file_content = re.sub(r"#.*", "", file_content)  # Remove everything after '#'
    file_content = re.sub(r"\s*\n\s*\n+", "\n", file_content)  # Remove extra newlines
    lines = file_content.splitlines()

    # OFF header file can be in two formats:
    #
    # OFF           <- Mark the start of the file
    # 8 6 12        <- Number of vertices, faces, and edges
    # ...
    #
    # or
    #
    # OFF8 6 12     <- Same as above but in one line
    # ...

    if len(lines) > 0 and lines[0] == "OFF":
        lines = lines[1:]
    elif len(lines) > 0 and lines[0].startswith("OFF") and len(lines[0]) > 3:
        lines[0] = lines[0][3:]

    if not lines:
        raise ValueError("OFF file is empty")

    # Parse number of vertices, faces, and edges (metadata)
    num_nodes, num_faces, *_ = map(int, lines[0].split())

    # Load nodes (vertices) using numpy
    nodes = np.array([list(map(float, line.split())) for line in lines[1 : 1 + num_nodes]])

    faces = []
    for line in lines[1 + num_nodes : 1 + num_nodes + num_faces]:
        face_data = list(map(int, line.split()))
        face_type = face_data[0]
        face_vertices = face_data[1:]

        if face_type == 3:
            faces.append(face_vertices)
        elif face_type == 4:
            faces.append([face_vertices[0], face_vertices[1], face_vertices[2]])
            faces.append([face_vertices[0], face_vertices[2], face_vertices[3]])

    return torch.from_numpy(nodes), torch.tensor(faces, dtype=torch.int64)
