import types
from typing import Any

import pytest

from torch_pointcloud.utils.deprecation import renamed_attribute


def _module_with_renamed_function() -> types.ModuleType:
    module = types.ModuleType("example")

    def new_function() -> int:
        return 1

    def module_getattr(name: str) -> Any:
        return vars(module)[renamed_attribute(name, {"old_function": "new_function"}, module.__name__)]

    module.new_function = new_function  # type: ignore[attr-defined]
    module.__getattr__ = module_getattr  # type: ignore[method-assign]
    return module


def test_renamed_attribute_resolves_the_former_name_with_a_warning_at_the_caller() -> None:
    module = _module_with_renamed_function()
    with pytest.warns(FutureWarning, match="`old_function` is deprecated; use `new_function` instead") as record:
        function = module.old_function
    assert function is module.new_function
    assert record[0].filename == __file__


def test_renamed_attribute_raises_attribute_error_for_an_unknown_name() -> None:
    module = _module_with_renamed_function()
    with pytest.raises(AttributeError, match="module 'example' has no attribute 'missing'"):
        module.missing  # noqa: B018
