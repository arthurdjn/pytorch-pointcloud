import functools
from typing import Any, Callable, Dict, Literal, Optional, Protocol, TypedDict, overload

from torch import nn

ModelCategory = Literal["unspecified", "classification", "segmentation", "detection"]


class ModelDict(TypedDict):
    name: str
    weights: Dict[str, Any]
    transforms: Dict[str, Any]
    params: Dict[str, Any]
    fn: Callable


_REGISTERED_MODELS: Dict[ModelCategory, Dict[str, ModelDict]] = {
    "unspecified": {},
    "classification": {},
    "segmentation": {},
    "detection": {},
}


class ClassificationModel(Protocol):
    def __init__(self, in_channels: int, num_classes: int, **kwargs: Any) -> None: ...

    def reset_classifier(self, num_classes: int, global_pool: Any, **kwargs: Any) -> None: ...

    def forward_features(self, *_: Any, **__: Any) -> Any: ...

    def forward_head(self, *_: Any, **__: Any) -> Any: ...

    def forward(self, *_: Any, **__: Any) -> Any: ...

    def __call__(self, *_: Any, **__: Any) -> Any: ...


class SegmentationModel(Protocol):
    def __init__(self, in_channels: int, num_classes: int, **kwargs: Any) -> None: ...

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None: ...

    def forward_features(self, *_: Any, **__: Any) -> Any: ...

    def forward_decoder(self, *_: Any, **__: Any) -> Any: ...

    def forward_head(self, *_: Any, **__: Any) -> Any: ...

    def forward(self, *_: Any, **__: Any) -> Any: ...

    def __call__(self, *_: Any, **__: Any) -> Any: ...


class DetectionModel(Protocol):
    def __init__(self, in_channels: int, num_classes: int, **kwargs: Any) -> None: ...

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None: ...

    def forward_features(self, *_: Any, **__: Any) -> Any: ...

    def forward_head(self, *_: Any, **__: Any) -> Any: ...

    def forward(self, *_: Any, **__: Any) -> Any: ...

    def __call__(self, *_: Any, **__: Any) -> Any: ...


@overload
def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    category: Literal["unspecified"],
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]: ...


@overload
def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    category: Literal["classification"],
) -> Callable[[Callable[..., ClassificationModel]], Callable[..., ClassificationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    category: Literal["segmentation"],
) -> Callable[[Callable[..., SegmentationModel]], Callable[..., SegmentationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    category: Literal["detection"],
) -> Callable[[Callable[..., DetectionModel]], Callable[..., DetectionModel]]: ...


def register_model(
    name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    category: ModelCategory = "unspecified",
) -> Callable:
    params = params or {}
    transforms = transforms or {}
    weights = weights or {}

    if category not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model category {category!r}. Expected one of: {expected_tasks}.")

    def decorator(fn: Callable) -> Callable:
        _REGISTERED_MODELS[category][name] = {
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
def create_model(name: str, category: Literal["unspecified"], *args: Any, **kwargs: Any) -> nn.Module: ...


@overload
def create_model(name: str, category: Literal["classification"], *args: Any, **kwargs: Any) -> ClassificationModel: ...


@overload
def create_model(name: str, category: Literal["segmentation"], *args: Any, **kwargs: Any) -> SegmentationModel: ...


@overload
def create_model(name: str, category: Literal["detection"], *args: Any, **kwargs: Any) -> DetectionModel: ...


def create_model(name: str, *args: Any, category: ModelCategory = "unspecified", **kwargs: Any) -> Any:
    if category not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model category {category!r}. Expected one of: {expected_tasks}.")

    if category == "unspecified":
        for category in _REGISTERED_MODELS.keys():
            model_dict = _REGISTERED_MODELS[category].get(name)
            if model_dict is not None:
                break
    else:
        model_dict = _REGISTERED_MODELS[category].get(name)

    if model_dict is None:
        available_models = ", ".join(f"{m!r}" for m in _REGISTERED_MODELS[category].keys())
        raise ValueError(f"Model {name!r} not found in {category!r} registry. Available models: {available_models}.")

    model_fn = model_dict["fn"]
    return model_fn(*args, **kwargs)
