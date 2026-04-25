import fnmatch
import functools
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, overload
from urllib.parse import urlparse

import torch
from torch import nn

from torch_pointcloud.config import MODELS_DIR

from ._base import BaseModel, ClassificationModel, DetectionModel, SegmentationModel

Task = Literal["base", "classification", "segmentation", "detection"]


class ModelDict(TypedDict):
    name: str
    weights: Optional[str]
    transforms: Optional[Callable]
    hparams: Dict[str, Any]
    fn: Callable


_REGISTERED_MODELS: Dict[Task, Dict[str, ModelDict]] = {
    "base": {},
    "classification": {},
    "segmentation": {},
    "detection": {},
}


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transforms: Optional[Callable] = None,
    weights: Optional[str] = None,
    task: Literal["base"],
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transforms: Optional[Callable] = None,
    weights: Optional[str] = None,
    task: Literal["classification"],
) -> Callable[[Callable[..., ClassificationModel]], Callable[..., ClassificationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transforms: Optional[Callable] = None,
    weights: Optional[str] = None,
    task: Literal["segmentation"],
) -> Callable[[Callable[..., SegmentationModel]], Callable[..., SegmentationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transforms: Optional[Callable] = None,
    weights: Optional[str] = None,
    task: Literal["detection"],
) -> Callable[[Callable[..., DetectionModel]], Callable[..., DetectionModel]]: ...


def register_model(
    name: str,
    *,
    task: Task,
    hparams: Optional[Dict[str, Any]] = None,
    transforms: Optional[Callable] = None,
    weights: Optional[str] = None,
) -> Callable:
    hparams = hparams or {}

    if task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    def decorator(fn: Callable) -> Callable:
        _REGISTERED_MODELS[task][name] = {
            "name": name,
            "transforms": transforms,
            "hparams": hparams,
            "weights": weights,
            "fn": fn,
        }

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> nn.Module:
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@overload
def create_model(
    name: str, task: Literal["base"], *, pretrained: bool = False, return_info: Literal[True], **kwargs: Any
) -> tuple[BaseModel, Dict[str, Any]]: ...


@overload
def create_model(
    name: str, task: Literal["base"], *, pretrained: bool = False, return_info: Literal[False] = False, **kwargs: Any
) -> BaseModel: ...


@overload
def create_model(
    name: str, task: Literal["classification"], *, pretrained: bool = False, return_info: Literal[True], **kwargs: Any
) -> tuple[ClassificationModel, Dict[str, Any]]: ...


@overload
def create_model(
    name: str,
    task: Literal["classification"],
    *,
    pretrained: bool = False,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> ClassificationModel: ...


@overload
def create_model(
    name: str, task: Literal["segmentation"], *, pretrained: bool = False, return_info: Literal[True], **kwargs: Any
) -> tuple[SegmentationModel, Dict[str, Any]]: ...


@overload
def create_model(
    name: str,
    task: Literal["segmentation"],
    *,
    pretrained: bool = False,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> SegmentationModel: ...


@overload
def create_model(
    name: str, task: Literal["detection"], *, pretrained: bool = False, return_info: Literal[True], **kwargs: Any
) -> tuple[DetectionModel, Dict[str, Any]]: ...


@overload
def create_model(
    name: str,
    task: Literal["detection"],
    *,
    pretrained: bool = False,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> DetectionModel: ...


@overload
def create_model(
    name: str, task: Task, *, pretrained: bool = False, return_info: bool = False, **kwargs: Any
) -> Any: ...


def create_model(name: str, task: Task, *, pretrained: bool = False, return_info: bool = False, **kwargs: Any) -> Any:
    if task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    model_info = _REGISTERED_MODELS[task].get(name)
    if model_info is None:
        available_models = ", ".join(f"{m!r}" for m in _REGISTERED_MODELS[task].keys())
        raise ValueError(f"Model {name!r} not found in {task!r} registry. Available models: {available_models}.")

    # create a copy of the model entry to avoid modifying the original entry stored in the registry
    model_info = model_info.copy()
    # the fn key is dropped and is not returned with the model info in case `return_info` is True
    model_fn = model_info.pop("fn")  # type: ignore[misc]

    kwargs = {**model_info["hparams"], **kwargs}
    model = model_fn(**kwargs)
    if not pretrained:
        if return_info:
            return model, model_info
        return model

    weights_path = model_info["weights"]
    if weights_path is None:
        warnings.warn(f"No pretrained weights available for model {name!r}.")
        return model

    parsed = urlparse(weights_path)
    local_path = Path(MODELS_DIR, parsed.path.lstrip("/"))
    if not local_path.exists():
        raise FileNotFoundError(f"Model weights not found at {local_path.as_posix()}. Download the weights first.")

    weights_data = torch.load(local_path, weights_only=True)
    state_dict = weights_data["state_dict"] if "state_dict" in weights_data else weights_data
    msg = model.load_state_dict(state_dict, strict=True)
    if msg.missing_keys or msg.unexpected_keys:
        warnings.warn(
            f"Model {name!r} loaded with unexpected.\n"
            f"Missing keys: {msg.missing_keys}\n"
            f"Unexpected keys: {msg.unexpected_keys}"
        )

    if return_info:
        return model, model_info
    return model


def list_models(name: str = "*", *, task: Task) -> List[str]:
    if task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    model_names = list(_REGISTERED_MODELS[task].keys())
    return sorted(fnmatch.filter(model_names, name))
