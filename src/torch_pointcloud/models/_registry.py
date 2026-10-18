"""The `create_model` factory, `register_model` and `deprecated_model` decorators, and pretrained weight registry."""

import difflib
import fnmatch
import warnings
from http.client import HTTPException
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, TypeVar, Union, overload
from urllib.error import HTTPError
from urllib.parse import urlparse

from packaging.version import InvalidVersion, Version
from torch import nn
from typing_extensions import NotRequired

from torch_pointcloud import __version__
from torch_pointcloud.config import MODELS_DIR
from torch_pointcloud.datasets.utils import download_url
from torch_pointcloud.transforms.base import Transform
from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.hub import HF_ENDPOINT, HF_URL_RE, load_hub_token
from torch_pointcloud.utils.state_dict import load_state_dict, read_state_dict
from torch_pointcloud.utils.types import PathLike, ValueCollection

from ._base import ClassificationModel, DetectionModel, PartSegmentationModel, SemanticSegmentationModel

Task = Literal["pretraining", "classification", "semantic-segmentation", "part-segmentation", "detection"]

_INSTALLATION_URL = "https://pytorch-pointcloud.org/latest/installation/"
F = TypeVar("F", bound=Callable[..., Any])


class WeightsDict(TypedDict):
    """Structured description of a pretrained checkpoint.

    Attributes:
        url: Location of the weight file, mirroring the Hub path so that `hf://` stands for
            `https://huggingface.co/` (e.g. `hf://torch-pointcloud/pointnext-sm.scanobjectnn-hardest.openpoints/resolve/a98ba0b8f7c9615acf0f38a1a20c97bfb01c2e79/model.safetensors`).
            Several URLs are read in order into a single state dict (e.g. an encoder file and its head file). A file
            in a gated repository downloads with the Hub token of the environment (see `load_hub_token`).
        pretrained_filter_fn: Called as `pretrained_filter_fn(state_dict, model)` on the state dict read from `url`
            before it is loaded, to rename, drop or reorder the entries of a checkpoint saved with another parameter
            layout (e.g. the authors' original release).
        dataset: Benchmark the checkpoint was trained on (e.g. `scanobjectnn`, `s3dis-area5`).
        metrics: Scores measured with this package's benchmark scripts, keyed by metric name
            (e.g. `{"OA": 88.20}`, `{"mIoU": 0.7604}`).
        classes: Label names in prediction-channel order, so `classes[i]` names class $i$ of the head.
        author: Tag identifying who trained the original checkpoint (e.g. `openpcdet`, `fair`).
        license: License of the weight file (e.g. `CC-BY-NC-4.0`), when the source repository declares one.
    """

    url: ValueCollection[str]
    pretrained_filter_fn: NotRequired[Callable[[Dict[str, Any], nn.Module], Dict[str, Any]]]
    dataset: NotRequired[str]
    metrics: NotRequired[Dict[str, float]]
    classes: NotRequired[Sequence[str]]
    author: NotRequired[str]
    license: NotRequired[str]


class DeprecatedModelDict(TypedDict):
    """Former name of a registered model, declared with `deprecated_model`.

    Attributes:
        name: The former registry name.
        deprecated_in: Release that renamed the model (e.g. `0.0.17`).
        removed_in: Release that drops the former name, or `None` when it is not scheduled.
        message: Sentence appended to the warning the former name raises, or `None`.
    """

    name: str
    deprecated_in: str
    removed_in: Optional[str]
    message: Optional[str]


class ModelDict(TypedDict):
    """Registry entry of a registered model: its entry point and the metadata `create_model` returns."""

    name: str
    weights: Optional[WeightsDict]
    transform: Optional[Transform]
    input_keys: Tuple[str, ...]
    hparams: Dict[str, Any]
    deprecated_names: Tuple[DeprecatedModelDict, ...]
    fn: Callable


class ModelInfoDict(TypedDict):
    """Registry entry of a built model, returned by `create_model(..., return_info=True)`."""

    name: str
    task: Task
    weights: Optional[WeightsDict]
    transform: Optional[Transform]
    input_keys: Tuple[str, ...]
    hparams: Dict[str, Any]


_REGISTERED_MODELS: Dict[Task, Dict[str, ModelDict]] = {
    "pretraining": {},
    "classification": {},
    "semantic-segmentation": {},
    "part-segmentation": {},
    "detection": {},
}

_DEPRECATED_MODELS: Dict[Task, Dict[str, str]] = {task: {} for task in _REGISTERED_MODELS}


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Transform] = None,
    input_keys: Sequence[str] = ("x", "pos", "batch"),
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["pretraining"],
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Transform] = None,
    input_keys: Sequence[str] = ("x", "pos", "batch"),
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["classification"],
) -> Callable[[Callable[..., ClassificationModel]], Callable[..., ClassificationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Transform] = None,
    input_keys: Sequence[str] = ("x", "pos", "batch"),
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["semantic-segmentation"],
) -> Callable[[Callable[..., SemanticSegmentationModel]], Callable[..., SemanticSegmentationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Transform] = None,
    input_keys: Sequence[str] = ("x", "pos", "batch"),
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["part-segmentation"],
) -> Callable[[Callable[..., PartSegmentationModel]], Callable[..., PartSegmentationModel]]: ...


@overload
def register_model(
    name: str,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Transform] = None,
    input_keys: Sequence[str] = ("x", "pos", "batch"),
    weights: Union[str, WeightsDict, None] = None,
    task: Literal["detection"],
) -> Callable[[Callable[..., DetectionModel]], Callable[..., DetectionModel]]: ...


def register_model(
    name: str,
    *,
    task: Task,
    hparams: Optional[Dict[str, Any]] = None,
    transform: Optional[Transform] = None,
    input_keys: Sequence[str] = ("x", "pos", "batch"),
    weights: Union[str, WeightsDict, None] = None,
) -> Callable:
    """Register a model entry point under `name` for `task`.

    The decorated callable becomes reachable through `create_model`. A bare `weights` URL string is
    normalized to a `WeightsDict`, so registry consumers always see the structured form. A `deprecated_model`
    decorator placed above adds the former names of the entry.

    Args:
        name: Registry name, `<architecture>[.<dataset tag>]` (e.g. `pointnext-sm.scanobjectnn-hardest.openpoints`).
        task: Registry the model belongs to (`pretraining`, `classification`, `semantic-segmentation`,
            `part-segmentation` or `detection`).
        hparams: Default keyword arguments the entry point is called with; `create_model` kwargs override them.
        transform: Evaluation transform reproducing the preprocessing the weights were trained with.
        input_keys: Batch keys `forward` takes positionally, in order; a dotted key reads an attribute (e.g.
            `octree.depth`). `utils.data.select_inputs` gathers them from a collated batch.
        weights: Pretrained checkpoint, either a URL string or a `WeightsDict` with metadata.

    Returns:
        The decorator registering its target callable.

    Raises:
        ValueError: If `task` is unknown, or if `name` is the former name of another model of `task`.

    Registering a classification entry point with weight metadata:

    ```python
    from torch_pointcloud.models import PointNetClassification, WeightsDict, register_model


    @register_model(
        "pointnet-demo.scanobjectnn",
        task="classification",
        hparams=dict(in_channels=0, num_classes=15),
        weights=WeightsDict(
            url="hf://my-org/pointnet-demo.scanobjectnn/resolve/main/model.safetensors",
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
        if name in _DEPRECATED_MODELS[task]:
            raise ValueError(
                f"Model name {name!r} is a deprecated name of {_DEPRECATED_MODELS[task][name]!r} for task {task!r}."
            )

        previous = _REGISTERED_MODELS[task].get(name)
        if previous is not None:
            warnings.warn(
                f"Model {name!r} is already registered for task {task!r}; overwriting the existing entry.",
                stacklevel=2,
            )
            for deprecation in previous["deprecated_names"]:
                _DEPRECATED_MODELS[task].pop(deprecation["name"], None)

        _REGISTERED_MODELS[task][name] = {
            "name": name,
            "transform": transform,
            "input_keys": tuple(input_keys),
            "hparams": hparams,
            "weights": weights_dict,
            "deprecated_names": (),
            "fn": fn,
        }
        return fn

    return decorator


def deprecated_model(
    name: str,
    *,
    deprecated_in: str,
    removed_in: Optional[str] = None,
    message: Optional[str] = None,
) -> Callable[[F], F]:
    """Keep a renamed model reachable under its former `name`, placed above its `register_model` decorator.

    `create_model` and `model_info` resolve `name` to the registered name under its task with a `FutureWarning`;
    once the package reaches `removed_in` they raise a `ValueError` naming the new name instead. `list_models` never
    lists former names, and a model of the same name registered under another task is not affected. Stack one
    decorator per former name.

    Args:
        name: Former registry name of the entry, under the task it is registered for.
        deprecated_in: Release that renamed the model (e.g. `0.0.17`).
        removed_in: Release that drops the former name; `None` leaves the removal unscheduled.
        message: Sentence appended to the warning, e.g. what else changed with the rename.

    Returns:
        The decorator adding the former name to the registry entry of its target callable.

    Raises:
        ValueError: If `deprecated_in` or `removed_in` is not a version string, if `removed_in` is not a later release
            than `deprecated_in`, if the decorated callable is not registered under exactly one name (the decorator
            belongs above `register_model`), or if `name` is the registered name itself, a registered model of its
            task or already a former name under its task.

    Renaming a registered model while keeping its former name loadable for one release:

    ```python
    from torch_pointcloud.models import PointNetClassification, deprecated_model, register_model


    @deprecated_model("pointnet.scanobjectnn", deprecated_in="0.0.17", removed_in="0.0.18")
    @register_model("pointnet-base.scanobjectnn", task="classification", hparams=dict(in_channels=0, num_classes=15))
    def pointnet_base(**hparams):
        return PointNetClassification(**hparams)
    ```
    """
    deprecated_version = _parse_version(deprecated_in, "deprecated_in")
    if removed_in is not None and _parse_version(removed_in, "removed_in") <= deprecated_version:
        raise ValueError(
            f"Model {name!r} is removed in {removed_in!r}, which is not a later release than its deprecation in "
            f"{deprecated_in!r}."
        )

    def decorator(fn: F) -> F:
        registered = [
            (task, entry)
            for task, entries in _REGISTERED_MODELS.items()
            for entry in entries.values()
            if entry["fn"] is fn
        ]
        if not registered:
            raise ValueError(
                f"Cannot declare the former name {name!r}: the entry point is not registered. Place `@deprecated_model` "
                "above `@register_model`."
            )
        if len(registered) > 1:
            names = ", ".join(f"{entry['name']!r} ({task!r})" for task, entry in registered)
            raise ValueError(f"Cannot declare the former name {name!r}: the entry point is registered as {names}.")

        task, entry = registered[0]
        if name == entry["name"]:
            raise ValueError(f"Deprecated model name {name!r} is the registered name itself.")
        if name in _REGISTERED_MODELS[task]:
            raise ValueError(f"Deprecated model name {name!r} is already a registered {task!r} model name.")
        if name in _DEPRECATED_MODELS[task]:
            raise ValueError(
                f"Deprecated model name {name!r} is already a deprecated name of {_DEPRECATED_MODELS[task][name]!r} "
                f"for task {task!r}."
            )

        deprecation: DeprecatedModelDict = {
            "name": name,
            "deprecated_in": deprecated_in,
            "removed_in": removed_in,
            "message": message,
        }
        entry["deprecated_names"] = (*entry["deprecated_names"], deprecation)
        _DEPRECATED_MODELS[task][name] = entry["name"]
        return fn

    return decorator


def _parse_version(version: str, argument: str) -> Version:
    """The parsed release `version` of a `deprecated_model` argument, raising a `ValueError` naming `argument`."""
    try:
        return Version(version)
    except InvalidVersion as err:
        raise ValueError(f"Expected a version string for `{argument}` (e.g. '0.0.17'), got {version!r}.") from err


def cache_path(url: str) -> Path:
    """Return the path a registered weight URL occupies in the models cache.

    A Hub URL `hf://<namespace>/<repo>/resolve/<revision>/<file>` maps to `MODELS_DIR/<repo>/<revision>/<file>`, so a
    checkpoint pinned to a new revision is downloaded again. Any other URL maps to `MODELS_DIR/<path>`.

    Args:
        url: One `url` of a registered `WeightsDict` (a `WeightsDict` listing several files names each one separately).

    Returns:
        Path of the weight file inside the models cache.

    Raises:
        TypeError: If `url` is not a single URL string.
        ValueError: If a `hf://` URL does not follow the Hub path layout.

    ```python
    from torch_pointcloud.models._registry import cache_path

    path = cache_path("hf://torch-pointcloud/pointnext-sm.scanobjectnn-hardest.openpoints/resolve/a98ba0b8f7c9615acf0f38a1a20c97bfb01c2e79/model.safetensors")
    ```
    """
    if not isinstance(url, str):
        raise TypeError(
            f"Expected a single weight URL string, got {type(url).__name__}; a `WeightsDict` may list several files "
            "in its `url`, pass them one at a time."
        )
    if url.startswith("hf://"):
        match = HF_URL_RE.match(url)
        if match is None:
            raise ValueError(
                f"Expected a Hub URL of the form hf://<namespace>/<repo>/resolve/<revision>/<file>, got {url!r}."
            )

        return Path(MODELS_DIR, match["repo"], match["revision"], match["file"])
    return Path(MODELS_DIR, urlparse(url).path.lstrip("/"))


def resolve_weights(name: str, url: str) -> Path:
    """Return the local cache path of a registered weight file, downloading it on first use.

    A `hf://` URL mirrors the Hub path, so it is fetched by replacing the scheme with `https://huggingface.co/`
    when the file is absent from the cache, with the Hub token of the environment (`load_hub_token`) when there is one.
    Any other scheme is expected to be placed in the cache by hand.

    Args:
        name: Registered model name, used in error messages.
        url: One `url` of the registered `WeightsDict` (a `WeightsDict` listing several files names each one separately).

    Returns:
        Path of the weight file inside the models cache.

    Raises:
        TypeError: If `url` is not a single URL string.
        ValueError: If the URL path escapes the models cache directory.
        FileNotFoundError: If the file is absent and cannot be downloaded.

    ```python
    from torch_pointcloud.models._registry import resolve_weights

    path = resolve_weights(
        "pointnext-sm.scanobjectnn-hardest.openpoints",
        "hf://torch-pointcloud/pointnext-sm.scanobjectnn-hardest.openpoints/resolve/a98ba0b8f7c9615acf0f38a1a20c97bfb01c2e79/model.safetensors",
    )
    ```
    """
    local_path = cache_path(url)
    if not local_path.resolve().is_relative_to(Path(MODELS_DIR).resolve()):
        raise ValueError(
            f"Cannot load weights for {name!r}: the local cache path derived from {url!r} resolves to "
            f"{local_path.resolve().as_posix()}, outside the models cache directory ({Path(MODELS_DIR).as_posix()})."
        )
    if local_path.exists():
        return local_path

    if not url.startswith("hf://"):
        raise FileNotFoundError(
            f"Model weights for {name!r} not found at {local_path.as_posix()}. Download the weights at {url!r} and "
            f"place them in the models cache directory ({Path(MODELS_DIR).as_posix()})."
        )

    file_url = HF_ENDPOINT + "/" + url[len("hf://") :]
    token = load_hub_token()
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        download_url(file_url, local_path, description=f"Downloading {local_path.name}", headers=headers)
    except (OSError, HTTPException) as e:
        if isinstance(e, HTTPError) and e.code in (401, 403):
            repo_url = file_url.split("/resolve/", 1)[0]
            raise FileNotFoundError(
                f"Could not download weights for {name!r} from {file_url}: {e}. The repository is gated: accept "
                f"its terms at {repo_url} and log in with `hf auth login` or set `HF_TOKEN`, or download the file "
                f"manually and place it at {local_path.as_posix()}."
            ) from e
        raise FileNotFoundError(
            f"Could not download weights for {name!r} from {file_url}: {e}. Download the file manually and "
            f"place it at {local_path.as_posix()}."
        ) from e

    return local_path


def _is_non_commercial_license(license: Optional[str]) -> bool:
    """Whether a license identifier restricts the weights to non-commercial use (e.g. `CC-BY-NC-4.0`)."""
    return license is not None and "-NC" in license.upper()


def _resolve_entry(name: str, task: Optional[Task]) -> Tuple[str, Task, ModelDict]:
    """The registered name, task and entry of `name`, with its task inferred when the name is registered under a single
    one. A former name of a model renamed under that task resolves to its new name with a `FutureWarning`, or raises
    once the package reaches the release that removes it."""
    if task is not None and task not in _REGISTERED_MODELS.keys():
        expected_tasks = ", ".join(f"{t!r}" for t in _REGISTERED_MODELS.keys())
        raise ValueError(f"Invalid model task {task!r}. Expected one of: {expected_tasks}.")

    if task is None:
        tasks = [t for t in _REGISTERED_MODELS if name in _REGISTERED_MODELS[t] or name in _DEPRECATED_MODELS[t]]
        if len(tasks) > 1:
            registered = " and ".join(f"{t!r}" for t in tasks)
            raise ValueError(f"Model {name!r} is registered under tasks {registered}; pass `task=` to pick one.")

        if not tasks:
            message = f"Model {name!r} is not registered."
            matches = difflib.get_close_matches(name, list_models(), n=3)
            if matches:
                message += " Did you mean " + " or ".join(f"{m!r}" for m in matches) + "?"
            message += " Use `list_models()` to list the registered names."
            raise ValueError(message)

        task = tasks[0]

    new = _DEPRECATED_MODELS[task].get(name)
    renamed_entry = _REGISTERED_MODELS[task].get(new) if new is not None else None
    if new is not None and renamed_entry is not None:
        deprecation = next(d for d in renamed_entry["deprecated_names"] if d["name"] == name)
        removed_in = deprecation["removed_in"]
        suffix = f" {deprecation['message']}" if deprecation["message"] else ""
        if removed_in is not None and Version(__version__) >= Version(removed_in):
            raise ValueError(
                f"Model {name!r} was renamed to {new!r} in {deprecation['deprecated_in']} and the former name was "
                f"removed in {removed_in}; use {new!r}.{suffix}"
            )

        warnings.warn(
            f"Model {name!r} was renamed to {new!r} in {deprecation['deprecated_in']}; the former name will be "
            f"removed in {removed_in or 'a future release'}.{suffix}",
            FutureWarning,
            stacklevel=3,
        )
        name = new

    entry = _REGISTERED_MODELS[task].get(name)
    if entry is None:
        message = f"Model {name!r} not found in the {task!r} registry."
        for other in _REGISTERED_MODELS:
            if other == task:
                continue
            if name in _REGISTERED_MODELS[other]:
                message += f" Model {name!r} is registered under task {other!r}; pass task={other!r}."
            elif name in _DEPRECATED_MODELS[other]:
                renamed = _DEPRECATED_MODELS[other][name]
                message += f" Model {name!r} was renamed to {renamed!r} under task {other!r}; pass task={other!r}."
        matches = difflib.get_close_matches(name, _REGISTERED_MODELS[task], n=3)
        if matches:
            message += " Did you mean " + " or ".join(f"{m!r}" for m in matches) + "?"
        message += f" Use `list_models(task={task!r})` to list the registered names."
        raise ValueError(message)

    return name, task, entry


def model_info(name: str, task: Optional[Task] = None) -> ModelInfoDict:
    """The registry entry of a model without building it: its task, weights, transform, input keys and hparams.

    Args:
        name: Registered model name (see `list_models`).
        task: Registry the model belongs to; optional when `name` is registered under a single task.

    Returns:
        The entry as a `ModelInfoDict`, with the registered `hparams`.

    Raises:
        ValueError: If `task` or `name` is unknown, or if `task` is omitted for a name registered under several tasks.

    Example:
        ```pycon
        >>> info = model_info("pointnet.modelnet40")
        >>> info["task"], info["input_keys"]
        ('classification', ('x', 'pos', 'batch'))

        ```
    """
    name, task, entry = _resolve_entry(name, task)
    return {
        "name": name,
        "task": task,
        "weights": entry["weights"],
        "transform": entry["transform"],
        "input_keys": entry["input_keys"],
        "hparams": dict(entry["hparams"]),
    }


@overload
def create_model(
    name: str,
    task: Literal["pretraining"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> Tuple[nn.Module, ModelInfoDict]: ...


@overload
def create_model(
    name: str,
    task: Literal["pretraining"],
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
) -> Tuple[ClassificationModel, ModelInfoDict]: ...


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
    task: Literal["semantic-segmentation"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> Tuple[SemanticSegmentationModel, ModelInfoDict]: ...


@overload
def create_model(
    name: str,
    task: Literal["semantic-segmentation"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> SemanticSegmentationModel: ...


@overload
def create_model(
    name: str,
    task: Literal["part-segmentation"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> Tuple[PartSegmentationModel, ModelInfoDict]: ...


@overload
def create_model(
    name: str,
    task: Literal["part-segmentation"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> PartSegmentationModel: ...


@overload
def create_model(
    name: str,
    task: Literal["detection"],
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> Tuple[DetectionModel, ModelInfoDict]: ...


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
    task: None = None,
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[True],
    **kwargs: Any,
) -> Tuple[nn.Module, ModelInfoDict]: ...


@overload
def create_model(
    name: str,
    task: None = None,
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: Literal[False] = False,
    **kwargs: Any,
) -> nn.Module: ...


@overload
def create_model(
    name: str,
    task: Optional[Task] = None,
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: bool = False,
    **kwargs: Any,
) -> Any: ...


def create_model(
    name: str,
    task: Optional[Task] = None,
    *,
    pretrained: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    return_info: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a registered model, optionally loading pretrained or local checkpoint weights.

    Registered weights load into the registered configuration, then an overridden `num_classes` rebuilds the head
    at its fresh initialization (`0` drops it). Other overrides (e.g. `in_channels`) keep the matching weights, and
    the keys whose shape changed keep their initialization. Model keys missing from the checkpoint raise.

    Args:
        name: Registered model name (see `list_models`).
        task: Registry the model belongs to (`pretraining`, `classification`, `semantic-segmentation`,
            `part-segmentation` or `detection`). Optional when `name` is registered under a single task.
        pretrained: Load the registered pretrained weights, warning when their license is non-commercial. Mutually
            exclusive with `checkpoint_path`.
        checkpoint_path: Local checkpoint to load instead of the registered weights. Supports `torch.save`
            files, `.safetensors`, and Lightning checkpoints (the wrapped network is extracted).
        return_info: Also return the registry entry (`ModelInfoDict`), with the resolved `task` and the effective
            `hparams`.
        **kwargs: Overrides merged into the registered `hparams` and passed to the model constructor.

    Returns:
        The model, or a `(model, info)` tuple when `return_info` is true.

    Raises:
        ValueError: If `task` or `name` is unknown, if `task` is omitted for a name registered under several tasks,
            if both `pretrained` and `checkpoint_path` are passed, or if `pretrained` is requested for an entry that
            registers no weights.
        TypeError: If the entry registers architecture hparams only and the data-dependent arguments
            (typically `in_channels` and `num_classes`) are not passed.
        ImportError: If the model needs an optional dependency that is not installed or does not meet its version
            requirement; the original error is chained.
        FileNotFoundError: If the checkpoint file does not exist, or if the registered weights are absent from
            the models cache and cannot be downloaded.

    Building a registered model, overriding its registered hparams, and inspecting its registry entry:

    ```python
    import torch_pointcloud as tp

    model = tp.create_model("pointnet.modelnet40")
    model = tp.create_model("pointnet.modelnet40", num_classes=10)
    model, info = tp.create_model("pointnet.modelnet40", return_info=True)
    ```
    """
    if pretrained and checkpoint_path is not None:
        raise ValueError("'pretrained' and 'checkpoint_path' are mutually exclusive. Pass a single weight source.")

    name, task, entry = _resolve_entry(name, task)

    # The returned info carries the EFFECTIVE hparams (registry defaults updated by `kwargs`), so a
    # caller (e.g. a LightningModule) can log exactly what built the model.
    hparams = {**entry["hparams"], **kwargs}
    # Registered weights load into the registered head; an overridden `num_classes` rebuilds it afterwards.
    rebuild_head = pretrained and task not in ("detection", "pretraining") and "num_classes" in kwargs
    overrides = {key: value for key, value in kwargs.items() if key != "num_classes"} if rebuild_head else kwargs
    try:
        model = entry["fn"](**{**entry["hparams"], **overrides})
    except ImportError as err:
        raise ImportError(
            f"Model {name!r} needs an optional dependency that could not be imported: {err} See {_INSTALLATION_URL}."
        ) from err
    except TypeError as err:
        if "required positional argument" not in str(err) and "required keyword-only argument" not in str(err):
            raise

        raise TypeError(
            f"Model {name!r} registers architecture hparams only; pass the data-dependent arguments to "
            f"`create_model` (e.g. `in_channels=`, `num_classes=`). Original error: {err}"
        ) from err

    if pretrained:
        weights = entry["weights"]
        if weights is None:
            raise ValueError(
                f"No pretrained weights are registered for model {name!r}. Pass `pretrained=False` for a random "
                f"initialization, or use `list_models(task={task!r}, pretrained=True)` to list the models that ship "
                "weights."
            )
        license = weights.get("license")
        if _is_non_commercial_license(license):
            warnings.warn(
                f"Pretrained weights of {name!r} are released under {license} (non-commercial); check the license terms "
                "before using them.",
                stacklevel=2,
            )

        urls = ensure_tuple(weights["url"])
        state_dict: Dict[str, Any] = {}
        for url in urls:
            local_path = resolve_weights(name, url)
            file_state_dict = read_state_dict(local_path)
            duplicates = sorted(key for key in file_state_dict if key in state_dict)
            if duplicates:
                warnings.warn(
                    f"Weight file {url!r} of {name!r} overrides keys of the files before it: {', '.join(duplicates)}.",
                    stacklevel=2,
                )
            state_dict.update(file_state_dict)
        if "pretrained_filter_fn" in weights:
            state_dict = weights["pretrained_filter_fn"](state_dict, model)

        load_state_dict(model, state_dict, source=", ".join(urls))
        if rebuild_head and model.num_classes != kwargs["num_classes"]:
            model.reset_classifier(kwargs["num_classes"])
    elif checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path.as_posix()}.")

        load_state_dict(model, read_state_dict(path), source=path.as_posix())

    if return_info:
        info: ModelInfoDict = {
            "name": name,
            "task": task,
            "weights": entry["weights"],
            "transform": entry["transform"],
            "input_keys": entry["input_keys"],
            "hparams": hparams,
        }
        return model, info
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
    with_weights = tp.list_models(task="semantic-segmentation", pretrained=True)
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
