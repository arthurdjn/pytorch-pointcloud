import pytest

import torch_pointcloud as tpc


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("datasets"),
        pytest.param("inferers"),
        pytest.param("layers"),
        pytest.param("losses"),
        pytest.param("models"),
        pytest.param("transforms"),
        pytest.param("utils"),
    ],
)
def test_subpackage_exposed(name: str) -> None:
    module = getattr(tpc, name)
    assert module.__name__ == f"torch_pointcloud.{name}"
    assert name in tpc.__all__


def test_all_attributes_resolve() -> None:
    for name in tpc.__all__:
        getattr(tpc, name)


def test_factory_functions_exposed() -> None:
    assert callable(tpc.create_model)
    assert callable(tpc.list_models)
    assert callable(tpc.register_model)


def test_version_is_string() -> None:
    assert isinstance(tpc.__version__, str)
