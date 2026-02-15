from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

from torch_pointcloud.transforms.transforms import Transform
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import KeyCollection

from . import functional as F


class Transformd(Transform, metaclass=ABCMeta):
    """Base class for dictionary transforms.

    This class is used to define transforms that operate on a dictionary of data,
    and implement utility methods for key iteration and error handling.

    Args:
        keys: The keys to apply the transform to.
        allow_missing_keys: If ``True``, the transform will not raise an error if the keys are not present in the data.

    """

    def __init__(self, keys: Optional[KeyCollection] = None, allow_missing_keys: bool = False) -> None:
        self.keys = ensure_tuple(keys, none_as_empty=True)
        self.allow_missing_keys = allow_missing_keys

    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().__call__(data)


class RandomSampled(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.RandomSample`.

    Args:
        keys: The keys to sample from.
        num_samples: The number of values to sample.
        allow_missing_keys: If ``True``, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self, keys: KeyCollection, num_samples: int, seed: Optional[int] = None, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_samples = num_samples
        self.seed = seed

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.random_sampled(
            data,
            keys=self.keys,
            num_samples=self.num_samples,
            seed=self.seed,
            allow_missing_keys=self.allow_missing_keys,
        )


class RandomSampleFaceVerticesd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.RandomSampleFaceVertices`.

    Args:
        keys: The keys to sample from.
        face_keys: The keys to sample the faces from.
        num_samples: The number of vertices to sample.
        include_normals: If ``True``, the normals will be included in the output.
        normals_key: The key to store the normals in.
        seed: The seed for the random number generator.
        allow_missing_keys: If ``True``, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        face_keys: KeyCollection,
        num_samples: int,
        include_normals: bool = True,
        normals_key: str = "normals",
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.face_keys = ensure_tuple_size(face_keys, len(self.keys))
        self.num_samples = num_samples
        self.include_normals = include_normals
        self.normals_key = normals_key
        self.seed = seed

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.random_sample_face_verticesd(
            data,
            keys=self.keys,
            face_keys=self.face_keys,
            num_samples=self.num_samples,
            include_normals=self.include_normals,
            normals_key=self.normals_key,
            seed=self.seed,
            allow_missing_keys=self.allow_missing_keys,
        )


class SampleFarthestPointsd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.SampleFarthestPoints`.

    Args:
        pos_key: The key to store the positions in.
        keys: The keys to sample the farthest points from.
        num_samples: The number of points to sample.
        ratio: The ratio of points to sample.
        random_start: Whether to start the sampling from a random point.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        pos_key: str,
        keys: Optional[KeyCollection] = None,
        num_samples: Optional[int] = None,
        ratio: Optional[float] = None,
        random_start: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.pos_key = pos_key
        self.num_samples = num_samples
        self.ratio = ratio
        self.random_start = random_start

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.sample_farthest_pointsd(
            data,
            pos_key=self.pos_key,
            keys=self.keys,
            num_samples=self.num_samples,
            ratio=self.ratio,
            random_start=self.random_start,
            allow_missing_keys=self.allow_missing_keys,
        )


class NormalizeScaled(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.NormalizeScale`.

    Args:
        keys: The keys to normalize the scale of.
        allow_missing_keys: If ``True``, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(self, keys: KeyCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.normalize_scaled(
            data,
            keys=self.keys,
            allow_missing_keys=self.allow_missing_keys,
        )


class RemoveNearOrigind(Transformd):
    """Dict-based class transform of `torch_pointcloud.transforms.dictionary.functional.remove_near_origind`.
    This transform is designed to remove points that are within a given radius of the origin.

    Args:
        pos_key: The key containing the positions / coordinates, used to compute the distance from the origin.
        keys: The keys to remove the near origin points from.
        radius: The radius of the sphere.
        return_mask: Whether to return the mask of the points removed.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        pos_key: str,
        keys: Optional[KeyCollection] = None,
        radius: float = 1e-3,
        return_mask: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.pos_key = pos_key
        self.radius = radius
        self.return_mask = return_mask

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.remove_near_origind(
            data,
            pos_key=self.pos_key,
            keys=self.keys,
            radius=self.radius,
            allow_missing_keys=self.allow_missing_keys,
        )


class Absd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.Absolute`.

    Args:
        keys: The keys to make the tensor absolute.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.dictionary.transforms import Absd
        >>> data = {"pos": torch.tensor([-1.0, 2.0, -3.0])}
        >>> transform = Absd(keys=["pos"])
        >>> transform(data)
        {"pos": tensor([1.0, 2.0, 3.0])}
    """

    def __init__(self, keys: KeyCollection, inplace: bool = False, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.inplace = inplace

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.absd(
            data,
            keys=self.keys,
            inplace=self.inplace,
            allow_missing_keys=self.allow_missing_keys,
        )


class BoundingBoxd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.BoundingBox`.

    Args:
        keys: The keys to compute the bounding box of.
        dst_keys: The keys to store the bounding box in.
        dim: The dimension to compute the bounding box over.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        dim: int = -1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = dst_keys
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.bounding_boxd(
            data,
            keys=self.keys,
            dst_keys=self.dst_keys,
            dim=self.dim,
            allow_missing_keys=self.allow_missing_keys,
        )


class InboxMaskd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.InboxMask`.

    Args:
        keys: The keys to create the mask for.
        bbox_key: The key to store the bounding box in.
        dst_keys: The keys to store the mask in.
        dim: The dimension to create the mask over.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.dictionary.transforms import InboxMaskd
        >>> data = {
        ...     "pos": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ...     "bbox": (0.0, 10.0, 0.0, 10.0, 0.0, 10.0),
        ... }
        >>> transform = InboxMaskd(keys=["pos"], bbox_key="bbox", dst_keys=["mask"])
        >>> transform(data)
        {"pos": tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
         "mask": tensor([[True, True, True], [True, True, True], [True, True, True]])}
    """

    def __init__(
        self,
        keys: KeyCollection,
        bbox_key: str,
        dst_keys: Optional[KeyCollection] = None,
        dim: int = -1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.bbox_key = bbox_key
        self.dst_keys = dst_keys
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.inbox_maskd(
            data,
            keys=self.keys,
            bbox_key=self.bbox_key,
            dst_keys=self.dst_keys,
            dim=self.dim,
            allow_missing_keys=self.allow_missing_keys,
        )


class ApplyMaskd(Transformd):
    """Class based variant of the dictionary transform `torch_pointcloud.transforms.dictionary.functional.apply_maskd`.
    This transform is designed to apply a mask to input tensors stored in a dictionary.

    Args:
        keys: The keys to apply the mask to.
        mask_key: The key to store the mask in.
        dst_keys: The keys to store the transformed data in.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.dictionary.transforms import ApplyMaskd
        >>> data = {"pos": torch.tensor([1.0, 2.0, 3.0]), "mask": torch.tensor([True, False, True])}
        >>> transform = ApplyMaskd(keys=["pos"], mask_key="mask")
        >>> transform(data)
        {"pos": tensor([1.0, 3.0])}
    """

    def __init__(
        self,
        keys: KeyCollection,
        mask_key: str,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.dst_keys = dst_keys

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.apply_maskd(
            data,
            keys=self.keys,
            mask_key=self.mask_key,
            dst_keys=self.dst_keys,
            allow_missing_keys=self.allow_missing_keys,
        )
