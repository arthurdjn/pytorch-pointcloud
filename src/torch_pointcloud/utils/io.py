"""File reading and writing for point cloud data: JSON, OFF meshes, PLY, and safetensors."""

import json
import re
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from numpy.lib.recfunctions import repack_fields
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .types import PathLike


def load_json(file_path: PathLike) -> Dict[str, Any]:
    """Read a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        The decoded JSON object.
    """
    with open(file_path) as f:
        return json.load(f)


def load_off(file_path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    r"""Read an OFF mesh, triangulating its quadrilateral faces.

    Args:
        file_path: Path to the OFF file.

    Returns:
        The vertices $(V, 3)$ and the triangle indices $(F, 3)$.
    """
    with open(file_path, "r") as f:
        file_content = f.read()

    # Use re to remove comments (both inline and full-line comments)
    file_content = re.sub(r"#.*", "", file_content)  # Remove everything after '#'
    file_content = re.sub(r"\s*\n\s*\n+", "\n", file_content)  # Remove extra newlines
    lines = file_content.splitlines()

    # OFF header file can be in two formats:
    #
    # OFF           <- Mark the start of the file
    # 8 6 12        <- Number of vertices, face, and edges
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

    # Parse number of vertices, face, and edges (metadata)
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

    return nodes, np.stack(faces)


_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_PLY_NAMES = {
    "i1": "char",
    "u1": "uchar",
    "i2": "short",
    "u2": "ushort",
    "i4": "int",
    "u4": "uint",
    "f4": "float",
    "f8": "double",
}
_PLY_BYTEORDERS = {"ascii": None, "binary_little_endian": "<", "binary_big_endian": ">"}

# A property is `(name, item dtype code, list count dtype code)`, the count code being `None` for a scalar property.
PlyProperty = Tuple[str, str, Optional[str]]
PlyElement = Tuple[str, int, List[PlyProperty]]


def load_ply(file_path: PathLike) -> Dict[str, np.ndarray]:
    r"""Read a PLY file into one structured array per element.

    Reads the ASCII and both binary encodings. A scalar property is a field of its element's array. A list property
    whose length is the same on every row is a field of shape $(k,)$; one whose length varies is an object field
    holding one array per row.

    Args:
        file_path: Path to the PLY file.

    Returns:
        The elements keyed by name, in file order, each a structured array with one field per property.
    """
    with open(file_path, "rb") as f:
        byteorder, elements = _read_ply_header(f)
        data: Dict[str, np.ndarray] = {}
        for name, count, properties in elements:
            if not properties:
                data[name] = np.empty(count, dtype=np.dtype([]))
            elif byteorder is None:
                data[name] = _read_ply_ascii_element(f, count, properties)
            else:
                data[name] = _read_ply_binary_element(f, count, properties, byteorder)
    return data


def save_ply(file_path: PathLike, elements: Mapping[str, np.ndarray], *, text: bool = False) -> None:
    r"""Write structured arrays as the elements of a PLY file.

    A scalar field becomes a scalar property and a field with a sub-array shape $(k,)$ a `list uchar` property of
    $k$ items. The binary encoding is little endian.

    Args:
        file_path: Path to the PLY file to write.
        elements: The elements keyed by name, written in that order, each a structured array.
        text: Write the ASCII encoding instead of the binary one.
    """
    header = ["ply", f"format {'ascii' if text else 'binary_little_endian'} 1.0"]
    for name, element in elements.items():
        if element.dtype.names is None:
            raise ValueError(f"PLY element {name!r} must be a structured array, got dtype {element.dtype}.")
        for label in (name, *element.dtype.names):
            if len(label.split()) != 1:
                raise ValueError(
                    f"PLY element and property names cannot be empty or contain whitespace, got {label!r}."
                )
        header.append(f"element {name} {len(element)}")
        for field in element.dtype.names:
            dtype = element.dtype[field]
            base = dtype.subdtype[0] if dtype.subdtype else dtype
            shape: Tuple[int, ...] = dtype.subdtype[1] if dtype.subdtype else ()
            code = base.str[1:]
            if code not in _PLY_NAMES:
                raise ValueError(f"PLY has no type for field {field!r} of dtype {base}.")
            if len(shape) == 0:
                header.append(f"property {_PLY_NAMES[code]} {field}")
            elif len(shape) == 1 and shape[0] <= 255:
                header.append(f"property list uchar {_PLY_NAMES[code]} {field}")
            else:
                raise ValueError(
                    f"PLY list field {field!r} must be a sub-array of at most 255 items, got shape {shape}."
                )
    header.append("end_header")

    with open(file_path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for element in elements.values():
            if text:
                f.write(_ply_ascii_rows(element).encode("ascii"))
            else:
                _ply_binary_rows(element).tofile(f)


def _ply_dtype(token: str) -> str:
    """The NumPy code of a PLY type name."""
    if token not in _PLY_DTYPES:
        raise ValueError(f"Unknown PLY property type {token!r}.")
    return _PLY_DTYPES[token]


def _read_ply_header(f: BinaryIO) -> Tuple[Optional[str], List[PlyElement]]:
    """Parse the header up to `end_header`: the byte order (`None` for ASCII) and the properties of each element."""
    if f.readline().strip() != b"ply":
        raise ValueError("Not a PLY file: the first line is not 'ply'.")
    encoding: Optional[str] = None
    elements: List[PlyElement] = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Truncated PLY header: no 'end_header' line.")
        tokens = line.decode("ascii", "replace").split()
        if not tokens or tokens[0] in ("comment", "obj_info"):
            continue
        if tokens[0] == "end_header":
            if encoding is None:
                raise ValueError("PLY header has no 'format' line.")
            return _PLY_BYTEORDERS[encoding], elements
        if tokens[0] == "format" and len(tokens) >= 2 and tokens[1] in _PLY_BYTEORDERS:
            encoding = tokens[1]
        elif tokens[0] == "element" and len(tokens) == 3:
            elements.append((tokens[1], int(tokens[2]), []))
        elif tokens[0] == "property" and len(tokens) == 5 and tokens[1] == "list" and elements:
            elements[-1][2].append((tokens[4], _ply_dtype(tokens[3]), _ply_dtype(tokens[2])))
        elif tokens[0] == "property" and len(tokens) == 3 and elements:
            elements[-1][2].append((tokens[2], _ply_dtype(tokens[1]), None))
        else:
            raise ValueError(f"Unexpected PLY header line {line.decode('ascii', 'replace').strip()!r}.")


def _read_exact(f: BinaryIO, size: int) -> bytes:
    """Read `size` bytes, refusing a short read."""
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Truncated PLY file: fewer bytes than the header declares.")
    return data


def _read_ply_binary_row(
    f: BinaryIO, properties: List[PlyProperty], byteorder: str
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Read one binary row property by property: the values, and the length of each list property."""
    values: Dict[str, Any] = {}
    lengths: Dict[str, int] = {}
    for name, code, count_code in properties:
        if count_code is None:
            dtype = np.dtype(byteorder + code)
            values[name] = np.frombuffer(_read_exact(f, dtype.itemsize), dtype=dtype)[0]
        else:
            count_dtype, item_dtype = np.dtype(byteorder + count_code), np.dtype(byteorder + code)
            length = int(np.frombuffer(_read_exact(f, count_dtype.itemsize), dtype=count_dtype)[0])
            items = np.frombuffer(_read_exact(f, length * item_dtype.itemsize), dtype=item_dtype)
            values[name] = items.astype(item_dtype.newbyteorder("="))
            lengths[name] = length
    return values, lengths


def _read_ply_binary_element(f: BinaryIO, count: int, properties: List[PlyProperty], byteorder: str) -> np.ndarray:
    """Read a binary element in one `fromfile` when every row has the layout of the first, row by row otherwise.

    The one-pass read is exact: the length of each list is checked at its true offset, so the first row that
    departs from the layout of the first row is caught before any later row is misread.
    """
    start = f.tell()
    lengths = _read_ply_binary_row(f, properties, byteorder)[1] if count else {n: 0 for n, _, c in properties if c}
    f.seek(start)

    fields: List[Any] = []
    for name, code, count_code in properties:
        if count_code is None:
            fields.append((name, byteorder + code))
        else:
            fields.append((f"{name} count", byteorder + count_code))
            fields.append((name, byteorder + code, (lengths[name],)))
    rows = np.fromfile(f, dtype=np.dtype(fields), count=count)
    if len(rows) == count and all((rows[f"{name} count"] == length).all() for name, length in lengths.items()):
        rows = repack_fields(rows[[name for name, _, _ in properties]])
        return rows if rows.dtype.isnative else rows.astype(rows.dtype.newbyteorder("="))

    f.seek(start)
    scalars: Dict[str, List[Any]] = {name: [] for name, _, count_code in properties if count_code is None}
    lists: Dict[str, List[np.ndarray]] = {name: [] for name, _, count_code in properties if count_code}
    for _ in range(count):
        values, _ = _read_ply_binary_row(f, properties, byteorder)
        for name, value in values.items():
            (lists if name in lists else scalars)[name].append(value)
    return _ply_element(count, properties, scalars, lists)


def _read_ply_ascii_element(f: BinaryIO, count: int, properties: List[PlyProperty]) -> np.ndarray:
    """Read an ASCII element, one row per line."""
    scalars: Dict[str, List[Any]] = {name: [] for name, _, count_code in properties if count_code is None}
    lists: Dict[str, List[np.ndarray]] = {name: [] for name, _, count_code in properties if count_code}
    for _ in range(count):
        line = f.readline()
        while line and not line.strip():
            line = f.readline()
        tokens = line.decode("ascii", "replace").split()
        if not tokens:
            raise ValueError("Truncated PLY file: fewer rows than the header declares.")
        position = 0
        for name, code, count_code in properties:
            if count_code is None:
                scalars[name].append(tokens[position])
                position += 1
            else:
                length = int(tokens[position])
                lists[name].append(np.array(tokens[position + 1 : position + 1 + length], dtype=code))
                position += 1 + length
    return _ply_element(count, properties, scalars, lists)


def _ply_element(
    count: int, properties: List[PlyProperty], scalars: Dict[str, List[Any]], lists: Dict[str, List[np.ndarray]]
) -> np.ndarray:
    """Pack per-property columns into a structured array; a list property of varying length becomes an object field."""
    fields: List[Any] = []
    for name, code, count_code in properties:
        if count_code is None:
            fields.append((name, code))
        else:
            lengths = {len(items) for items in lists[name]} or {0}
            fields.append((name, code, (lengths.pop(),)) if len(lengths) == 1 else (name, object))
    element = np.empty(count, dtype=np.dtype(fields))
    for name, code, count_code in properties:
        if count_code is None:
            element[name] = np.array(scalars[name], dtype=code)
        elif element.dtype[name].kind == "O":
            for index, items in enumerate(lists[name]):
                element[name][index] = items
        elif count:
            element[name] = np.stack(lists[name])
    return element


def _ply_binary_rows(element: np.ndarray) -> np.ndarray:
    """Lay an element out as little-endian rows, each list field preceded by its `uchar` length."""
    fields: List[Any] = []
    for field in element.dtype.names or ():
        base, shape = element.dtype[field].subdtype or (element.dtype[field], ())
        if shape:
            fields.append((f"{field} count", "u1"))
        fields.append((field, "<" + base.str[1:], shape))
    rows = np.empty(len(element), dtype=np.dtype(fields))
    for field in element.dtype.names or ():
        _, shape = element.dtype[field].subdtype or (element.dtype[field], ())
        if shape:
            rows[f"{field} count"] = shape[0]
        rows[field] = element[field]
    return rows


def _ply_ascii_token(value: Any) -> str:
    """A number as text: the shortest form that reads back to the same value of its own dtype."""
    return np.format_float_positional(value, trim="-") if isinstance(value, np.floating) else str(value)


def _ply_ascii_rows(element: np.ndarray) -> str:
    """Lay an element out as text rows, each list field preceded by its length."""
    lines = []
    for row in element:
        tokens = []
        for field in element.dtype.names or ():
            value = row[field]
            if isinstance(value, np.ndarray):
                tokens.append(str(len(value)))
                tokens.extend(_ply_ascii_token(item) for item in value)
            else:
                tokens.append(_ply_ascii_token(value))
        lines.append(" ".join(tokens))
    return "".join(line + "\n" for line in lines)


def save_safetensors(file_path: PathLike, data: Mapping[str, Any]) -> None:
    """Write a mapping to a safetensors file, storing its string entries as metadata.

    Args:
        file_path: Path to the safetensors file to write.
        data: Mapping of names to tensors, scalars (wrapped in a tensor) or strings.
    """
    data = {k: torch.tensor(v) if isinstance(v, (int, float, bool)) else v for k, v in data.items()}
    metadata = {k: v for k, v in data.items() if isinstance(v, str)}
    tensors = {k: v.contiguous() for k, v in data.items() if isinstance(v, Tensor)}
    save_file(tensors, file_path, metadata=metadata)


def load_safetensors(file_path: PathLike) -> Dict[str, Tensor | str]:
    """Read a safetensors file, merging its metadata back into the tensor mapping.

    Args:
        file_path: Path to the safetensors file.

    Returns:
        The file's tensors and string metadata, keyed by name.
    """
    with safe_open(file_path, framework="pt") as f:
        metadata = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return {**metadata, **tensors}
