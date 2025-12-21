import fnmatch
import functools
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, overload

from torch import nn

from ._base import ClassificationModel, DetectionModel, SegmentationModel

Task = Literal["unspecified", "classification", "segmentation", "detection"]


class ModelDict(TypedDict):
    name: str
    weights: Dict[str, Any]
    transforms: Dict[str, Any]
    params: Dict[str, Any]
    fn: Callable


_REGISTERED_MODELS: Dict[Task, Dict[str, ModelDict]] = {
    "unspecified": {},
    "classification": {},
    "segmentation": {},
    "detection": {},
}


@overload
def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    task: Literal["unspecified"],
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]: ...


@overload
def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    task: Literal["classification"],
) -> Callable[[Callable[..., ClassificationModel]], Callable[..., ClassificationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    task: Literal["segmentation"],
) -> Callable[[Callable[..., SegmentationModel]], Callable[..., SegmentationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    task: Literal["detection"],
) -> Callable[[Callable[..., DetectionModel]], Callable[..., DetectionModel]]: ...


def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    task: Task = "unspecified",
) -> Callable:
    params = params or {}
    transforms = transforms or {}
    weights = weights or {}

    if task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    def decorator(fn: Callable) -> Callable:
        _REGISTERED_MODELS[task][name] = {
            "name": name,
            "transforms": transforms,
            "params": params,
            "weights": weights,
            "fn": fn,
        }

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> nn.Module:
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@overload
def create_model(name: str, task: Literal["unspecified"], *args: Any, **kwargs: Any) -> nn.Module: ...


@overload
def create_model(name: str, task: Literal["classification"], *args: Any, **kwargs: Any) -> ClassificationModel: ...


@overload
def create_model(name: str, task: Literal["segmentation"], *args: Any, **kwargs: Any) -> SegmentationModel: ...


@overload
def create_model(name: str, task: Literal["detection"], *args: Any, **kwargs: Any) -> DetectionModel: ...


def create_model(name: str, *args: Any, task: Task = "unspecified", **kwargs: Any) -> Any:
    if task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    if task == "unspecified":
        for task in _REGISTERED_MODELS.keys():
            model_dict = _REGISTERED_MODELS[task].get(name)
            if model_dict is not None:
                break
    else:
        model_dict = _REGISTERED_MODELS[task].get(name)

    if model_dict is None:
        available_models = ", ".join(f"{m!r}" for m in _REGISTERED_MODELS[task].keys())
        raise ValueError(f"Model {name!r} not found in {task!r} registry. Available models: {available_models}.")

    model_fn = model_dict["fn"]
    return model_fn(*args, **kwargs)


def list_models(name: str = "*", task: Task = "unspecified") -> List[str]:
    if task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    model_names = []
    if task == "unspecified":
        for task in _REGISTERED_MODELS.keys():
            model_names.extend(list(_REGISTERED_MODELS[task].keys()))
    else:
        model_names.extend(list(_REGISTERED_MODELS[task].keys()))

    return sorted(fnmatch.filter(model_names, name))
