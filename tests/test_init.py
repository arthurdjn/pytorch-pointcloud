import pytest

import torch_pointcloud as tp


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
    module = getattr(tp, name)
    assert module.__name__ == f"torch_pointcloud.{name}"
    assert name in tp.__all__


def test_all_attributes_resolve() -> None:
    for name in tp.__all__:
        getattr(tp, name)


def test_factory_functions_exposed() -> None:
    assert callable(tp.create_model)
    assert callable(tp.list_models)
    assert callable(tp.register_model)


def test_version_is_string() -> None:
    assert isinstance(tp.__version__, str)
