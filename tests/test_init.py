import subprocess
import sys

import pytest

import torch_pointcloud as tp
from torch_pointcloud.utils.imports import _LIGHTNING_AVAILABLE


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("datasets"),
        pytest.param("inferers"),
        pytest.param("layers"),
        pytest.param("losses"),
        pytest.param("metrics"),
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
    assert callable(tp.model_info)
    assert callable(tp.register_model)


def test_version_is_string() -> None:
    assert isinstance(tp.__version__, str)


@pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed")
def test_lightning_subpackage_loads_on_first_access() -> None:
    code = (
        "import sys; import torch_pointcloud as tp; print('lightning' in sys.modules); "
        "print(tp.lightning.__name__); print('lightning' in sys.modules)"
    )
    stdout = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert stdout.split("\n")[:3] == ["False", "torch_pointcloud.lightning", "True"]
