import json
import re
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch

from .types import PATH_LIKE


def load_json(file_path: PATH_LIKE) -> Dict[str, Any]:
    with open(file_path) as f:
        return json.load(f)


def load_off(file_path: PATH_LIKE) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(file_path, "r") as f:
        file_content = f.read()

    # Use re to remove comments (both inline and full-line comments)
    file_content = re.sub(r"#.*", "", file_content)  # Remove everything after '#'
    file_content = re.sub(r"\s*\n\s*\n+", "\n", file_content)  # Remove extra newlines
    lines = file_content.splitlines()

    if len(lines) > 0 and lines[0] == "OFF":
        lines = lines[1:]

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

    return torch.tensor(nodes, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)


def save_off(file_path: PATH_LIKE, vertices: torch.Tensor, faces: Union[torch.Tensor, Sequence[torch.Tensor]]) -> None:
    """Saves a set of vertices and faces to an .off file.

    Args:
        file_path: The name of the file to save.
        vertices: A tensor of shape (N, 3) representing the vertices.
        faces: A tensor of shape (M, 3 or 4) representing the faces. Can be triangular or quadrilateral.
    """
    with open(file_path, "w") as f:
        f.write("OFF\n")
        f.write(f"{vertices.shape[0]} {len(faces)} 0\n")

        for vertex in vertices:
            f.write(f"{vertex[0].item()} {vertex[1].item()} {vertex[2].item()}\n")

        for face in faces:
            if face.shape[0] == 3:
                f.write(f"3 {face[0].item()} {face[1].item()} {face[2].item()}\n")
            elif face.shape[0] == 4:
                f.write(f"4 {face[0].item()} {face[1].item()} {face[2].item()} {face[3].item()}\n")
            else:
                raise ValueError(f"Unsupported face size: {face.shape[0]}")
