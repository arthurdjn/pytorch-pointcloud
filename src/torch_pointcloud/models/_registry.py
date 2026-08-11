import difflib
import fnmatch
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypedDict, Union, overload
from urllib.parse import urlparse

from torch import nn
from typing_extensions import NotRequired

from torch_pointcloud.config import MODELS_DIR
from torch_pointcloud.utils.state_dict import load_state_dict, read_state_dict
from torch_pointcloud.utils.types import PathLike

from ._base import ClassificationModel, DetectionModel, SegmentationModel

Task = Literal["base", "classification", "segmentation", "detection"]


class WeightsDict(TypedDict):
    """Structured description of a pretrained checkpoint.

    Attributes:
        url: Location of the weight file (e.g. `hf://torch-pointcloud/pointnext/pointnext-sm.scanobjectnn.openpoints.safetensors`).
        dataset: Benchmark the checkpoint was trained on (e.g. `scanobjectnn`, `s3dis-area5`).
        metrics: Scores measured with this package's benchmark scripts, keyed by metric name
            (e.g. `{"OA": 88.20}`, `{"mIoU": 0.7604}`).
        classes: Label names in prediction-channel order, so `classes[i]` names class $i$ of the head.
        author: Tag identifying who trained the original checkpoint (e.g. `openpcdet`, `facebookresearch`).
        license: License of the weight file (e.g. `CC-BY-NC-4.0`), when the source repository declares one.
    """

    url: str
    dataset: NotRequired[str]
    metrics: NotRequired[Dict[str, float]]
    classes: NotRequired[Sequence[str]]
    author: NotRequired[str]
    license: NotRequired[str]


class ModelDict(TypedDict):
    """Registry entry describing a registered model, returned by `create_model(..., return_info=True)`."""

    name: str
    weights: Optional[WeightsDict]
    transform: Optional[Callable]
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
    transform: Optional[Callable] = None,
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["base"],
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Callable] = None,
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["classification"],
) -> Callable[[Callable[..., ClassificationModel]], Callable[..., ClassificationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Callable] = None,
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["segmentation"],
) -> Callable[[Callable[..., SegmentationModel]], Callable[..., SegmentationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Callable] = None,
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["detection"],
) -> Callable[[Callable[..., DetectionModel]], Callable[..., DetectionModel]]: ...


def register_model(
    name: str,
    *,
    task: Task,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Callable] = None,
    weights: Union[str, WeightsDict, None] = None,
) -> Callable:
    """Register a model entry point under `name` for `task`.

    The decorated callable becomes reachable through `create_model`. A bare `weights` URL string is
    normalized to a `WeightsDict`, so registry consumers always see the structured form.

    Args:
        name: Registry name, `<architecture>[.<dataset tag>]` (e.g. `pointnext-sm.scanobjectnn.openpoints`).
        task: Registry the model belongs to (`base`, `classification`, `segmentation`, or `detection`).
        hparams: Default keyword arguments the entry point is called with; `create_model` kwargs override them.
        transform: Evaluation transform reproducing the preprocessing the weights were trained with.
        weights: Pretrained checkpoint, either a URL string or a `WeightsDict` with metadata.

    Returns:
        The decorator registering its target callable.

    Registering a classification entry point with weight metadata:

    ```python
    from torch_pointcloud.models import PointNetClassification, WeightsDict, register_model


    @register_model(
        "pointnet-demo.scanobjectnn",
        task="classification",
        hparams=dict(in_channels=0, num_classes=15),
        weights=WeightsDict(
            url="hf://my-org/pointnet/pointnet-demo.scanobjectnn.safetensors",
            dataset="scanobjectnn",
            metrics={"OA": 88.20},
        ),
    )
    def pointnet_demo(**hparams):
        return PointNetClassification(**hparams)
    ```
    """
    hparams = hparams or {}
    weights_dict: Optional[WeightsDict] = {"url": weights} if isinstance(weights, str) else weights

    if task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTERED_MODELS[task]:
            warnings.warn(
                f"Model {name!r} is already registered for task {task!r}; overwriting the existing entry.",
                stacklevel=2,
            )
        _REGISTERED_MODELS[task][name] = {
            "name": name,
            "transform": transform,
            "hparams": hparams,
            "weights": weights_dict,
            "fn": fn,
        }
        return fn

    return decorator


@overload
def create_model(
    name: str,
    task: Literal["base"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> tuple[nn.Module, Dict[str, Any]]: ...


@overload
def create_model(
    name: str,
    task: Literal["base"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> nn.Module: ...


@overload
def create_model(
    name: str,
    task: Literal["classification"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> tuple[ClassificationModel, Dict[str, Any]]: ...


@overload
def create_model(
    name: str,
    task: Literal["classification"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> ClassificationModel: ...


@overload
def create_model(
    name: str,
    task: Literal["segmentation"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> tuple[SegmentationModel, Dict[str, Any]]: ...


@overload
def create_model(
    name: str,
    task: Literal["segmentation"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> SegmentationModel: ...


@overload
def create_model(
    name: str,
    task: Literal["detection"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> tuple[DetectionModel, Dict[str, Any]]: ...


@overload
def create_model(
    name: str,
    task: Literal["detection"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> DetectionModel: ...


@overload
def create_model(
    name: str,
    task: Task,
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: bool = False,
    **kwargs: Any,
) -> Any: ...


def create_model(
    name: str,
    task: Task,
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a registered model, optionally loading pretrained or local checkpoint weights.

    Weights load through a head-adapting policy: overriding `num_classes` (or `in_channels`) keeps the
    matching backbone weights and leaves the rebuilt head at its fresh initialization with a warning, while
    an untouched configuration loads completely. Model keys missing from the checkpoint raise.

    Args:
        name: Registered model name (see `list_models`).
        task: Registry the model belongs to (`base`, `classification`, `segmentation`, or `detection`).
        pretrained: Load the registered pretrained weights. Mutually exclusive with `checkpoint_path`.
        checkpoint_path: Local checkpoint to load instead of the registered weights. Supports `torch.save`
            files, `.safetensors`, and Lightning checkpoints (the wrapped network is extracted).
        return_info: Also return the registry entry, with `hparams` reflecting the effective values.
        **kwargs: Overrides merged into the registered `hparams` and passed to the model constructor.

    Returns:
        The model, or a `(model, info)` tuple when `return_info` is true.

    Raises:
        ValueError: If `task` or `name` is unknown, or both `pretrained` and `checkpoint_path` are passed.
        FileNotFoundError: If the weight or checkpoint file does not exist.

    Building a registered model, overriding its registered hparams, and inspecting its registry entry:

    ```python
    import torch_pointcloud as tp

    model = tp.create_model("pointnet.modelnet40", task="classification")
    model = tp.create_model("pointnet.modelnet40", task="classification", num_classes=10)
    model, info = tp.create_model("pointnet.modelnet40", task="classification", return_info=True)
    ```
    """
    if task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")
    if pretrained and checkpoint_path is not None:
        raise ValueError("'pretrained' and 'checkpoint_path' are mutually exclusive. Pass a single weight source.")

    model_info = _REGISTERED_MODELS[task].get(name)
    if model_info is None:
        message = f"Model {name!r} not found in the {task!r} registry."
        matches = difflib.get_close_matches(name, _REGISTERED_MODELS[task], n=3)
        if matches:
            message += " Did you mean " + " or ".join(f"{m!r}" for m in matches) + "?"
        other_tasks = [t for t, entries in _REGISTERED_MODELS.items() if t != task and name in entries]
        if other_tasks:
            message += " The name is registered under task " + " and ".join(f"{t!r}" for t in other_tasks) + "."
        message += f" Use `list_models(task={task!r})` to list the registered names."
        raise ValueError(message)

    # create a copy of the model entry to avoid modifying the original entry stored in the registry
    model_info = model_info.copy()
    # the fn key is dropped and is not returned with the model info in case `return_info` is True
    model_fn = model_info.pop("fn")  # type: ignore[misc]

    # The returned info carries the EFFECTIVE hparams (registry defaults updated by `kwargs`), so a
    # caller (e.g. a LightningModule) can log exactly what built the model.
    hparams = {**model_info["hparams"], **kwargs}
    model_info["hparams"] = hparams
    model = model_fn(**hparams)

    if pretrained:
        weights = model_info["weights"]
        if weights is None:
            warnings.warn(f"No pretrained weights available for model {name!r}.", stacklevel=2)
        else:
            parsed = urlparse(weights["url"])
            local_path = Path(MODELS_DIR, parsed.path.lstrip("/"))
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Model weights not found at {local_path.as_posix()}. Download the weights first."
                )
            load_state_dict(model, read_state_dict(local_path), source=weights["url"])
    elif checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path.as_posix()}.")
        load_state_dict(model, read_state_dict(path), source=path.as_posix())

    if return_info:
        return model, model_info
    return model


def list_models(name: str = "*", *, task: Optional[Task] = None, pretrained: bool = False) -> List[str]:
    """List registered model names, sorted alphabetically.

    Args:
        name: Wildcard filter on the registered name (`fnmatch` syntax, e.g. `"pointnext*"`).
        task: Restrict the listing to one registry; `None` lists across all tasks (duplicates removed).
        pretrained: Only list models that ship pretrained weights.

    Returns:
        The matching model names.

    Filtering by name pattern, task, and weight availability:

    ```python
    import torch_pointcloud as tp

    classifiers = tp.list_models("pointnext*", task="classification")
    with_weights = tp.list_models(task="segmentation", pretrained=True)
    everything = tp.list_models()
    ```
    """
    if task is not None and task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    tasks = [task] if task is not None else list(_REGISTERED_MODELS.keys())
    model_names = {
        model_name
        for t in tasks
        for model_name, entry in _REGISTERED_MODELS[t].items()
        if not pretrained or entry["weights"] is not None
    }
    return sorted(fnmatch.filter(model_names, name))
