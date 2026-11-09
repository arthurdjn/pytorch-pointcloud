import importlib

import pytest

import torch_pointcloud.inferers as inferers


@pytest.mark.parametrize(
    "module,former_name,name",
    [
        pytest.param("sliding_window", "SlidingWindowInferer", "SlidingWindowSegmentationInferer"),
        pytest.param("sliding_window", "sliding_window_inference", "sliding_window_segmentation_inference"),
        pytest.param("knn_window", "KNNWindowInferer", "KNNWindowSegmentationInferer"),
        pytest.param("knn_window", "knn_window_inference", "knn_window_segmentation_inference"),
        pytest.param("potential_sphere", "PotentialSphereInferer", "PotentialSphereSegmentationInferer"),
        pytest.param("potential_sphere", "potential_sphere_inference", "potential_sphere_segmentation_inference"),
        pytest.param("voxel_partition", "VoxelPartitionInferer", "VoxelPartitionSegmentationInferer"),
        pytest.param("tta", "TTAInferer", "TTASegmentationInferer"),
    ],
)
def test_former_inferer_names_warn_and_resolve(module: str, former_name: str, name: str) -> None:
    submodule = importlib.import_module(f"torch_pointcloud.inferers.{module}")
    for namespace in (inferers, submodule):
        with pytest.warns(FutureWarning, match=f"`{former_name}` is deprecated; use `{name}` instead"):
            assert getattr(namespace, former_name) is getattr(submodule, name)
    assert former_name not in inferers.__all__


def test_unknown_inferer_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchInferer'"):
        inferers.NoSuchInferer  # noqa: B018
