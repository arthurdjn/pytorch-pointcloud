from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Sequence

from torch_geometric.data import Data

from . import functional as F


class BaseTransform(ABC):
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]: ...

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.transform(data)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class Compose(BaseTransform):
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data

    def extra_repr(self) -> str:
        repr_str = ""
        for t in self.transforms:
            repr_str += f"\n    {t}"
        repr_str += "\n"
        return repr_str


class SampleRandomPoints(BaseTransform):
    def __init__(self, num_points: int, keys: Sequence[str] = ("xyz",)) -> None:
        assert len(keys) > 0, "keys must be a non-empty list"
        self.num_points = num_points
        self.keys = keys

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.sample_random_points(data, self.num_points, self.keys)

    def extra_repr(self) -> str:
        return f"num_points={self.num_points}, keys={tuple(self.keys)}"


class SampleFurthestPoints(BaseTransform):
    def __init__(self, num_points: int, keys: Sequence[str] = ("xyz",)) -> None:
        self.num_points = num_points
        self.keys = keys

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.sample_furthest_points(data, self.num_points, self.keys)

    def extra_repr(self) -> str:
        return f"num_points={self.num_points}, keys={tuple(self.keys)}"


class SampleMeshPoints(BaseTransform):
    def __init__(self, num_points: int, include_normals: bool = True) -> None:
        self.num_points = num_points
        self.include_normals = include_normals

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.sample_mesh_points(data, self.num_points, self.include_normals)

    def extra_repr(self) -> str:
        return f"num_points={self.num_points}, include_normals={self.include_normals}"


class NormalizeScale(BaseTransform):
    def __init__(self, keys: Sequence[str] = ("xyz",)) -> None:
        self.keys = keys

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.normalize_scale(data, self.keys)

    def extra_repr(self) -> str:
        return f"keys={tuple(self.keys)}"


class ToTorchGeometricData(BaseTransform):
    def transform(self, data: Dict[str, Any]) -> Data:
        return Data(**data)
