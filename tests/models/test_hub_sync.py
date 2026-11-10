"""Check the Hugging Face Hub mirror of the registry.

Every checkpoint whose weights live in the organisation is one public `torch-pointcloud/<name>` repository holding
`model.safetensors`, `config.json` and a card. The registry is the source of truth, so the card metadata (license,
library, `model-index` metrics), the `config.json` (the registry entry as data) and the pinned weights must follow it.
The organisation may hold more repositories than the registry: the weights of a checkpoint are pushed before its
registry entry lands on main.

The tests run with `--run-hub` only: they query the Hub API, one request per repository and revision, with the token of
the environment when there is one. `config.json` is never fetched (its requests are what the Hub counts as downloads);
the served file is compared by git blob id with a fresh render.
"""

import hashlib
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import pytest

from torch_pointcloud.models._registry import _DEPRECATED_MODELS, _REGISTERED_MODELS, ModelDict, Task, WeightsDict
from torch_pointcloud.utils.hub import HF_ENDPOINT, HF_URL_RE, load_hub_token

pytestmark = pytest.mark.hub

HF_NAMESPACE = "torch-pointcloud"
FORMER_KEPT_REPOSITORIES: Dict[str, str] = {
    "sonata-lp.scannet20.fair": "sonata-base-lp.scannet20.fair",
    "utonia.pretrain.pointcept": "utonia-base.pretrain.pointcept",
    "utonia-lp.scannet20.pointcept": "utonia-base-lp.scannet20.pointcept",
}
"""Kept repositories named after a former registry name that no release resolves any more, with the checkpoint their
weights are registered under today."""


def hub_json(path: str) -> Any:
    """The JSON the Hub API answers at `api/<path>`, `None` for a missing repository or revision (the Hub answers 401
    instead of 404 for a missing or private repository when the request carries no token)."""
    token = load_hub_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(f"{HF_ENDPOINT}/api/{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (401, 404):
            return None
        raise


def hub_api(path: str) -> Optional[Dict[str, Any]]:
    """The record of one repository or revision, `None` when it is missing."""
    record: Optional[Dict[str, Any]] = hub_json(path)
    return record


def repository_files(info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The files of a repository at the queried revision, keyed by name (`blobId`, and `lfs.sha256` for a weight file)."""
    return {sibling["rfilename"]: sibling for sibling in info.get("siblings", [])}


def blob_id(data: bytes) -> str:
    """The git blob id of a file's content, what the Hub reports for a plain git file."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hosted_here(weights: WeightsDict) -> bool:
    """Whether the weights live in an organisation repository rather than in the authors' own one."""
    return isinstance(weights["url"], str) and weights["url"].startswith(f"hf://{HF_NAMESPACE}/")


def hosted_checkpoints() -> Dict[str, Tuple[Task, ModelDict]]:
    """Every registered checkpoint whose weights live in the organisation, with its task and registry entry."""
    checkpoints: Dict[str, Tuple[Task, ModelDict]] = {}
    for task, entries in _REGISTERED_MODELS.items():
        for name, entry in entries.items():
            if entry["weights"] is not None and hosted_here(entry["weights"]):
                checkpoints[name] = (task, entry)
    return dict(sorted(checkpoints.items()))


def kept_repositories() -> Dict[str, Tuple[Task, str, ModelDict]]:
    """The organisation repositories of the checkpoints that now load the authors' own file, under their registered and
    former names, with the task, registered name and entry their weights are registered under today. Released
    versions still load them, so they stay published and only their `config.json` follows the registry."""
    kept: Dict[str, Tuple[Task, str, ModelDict]] = {}
    for task, entries in _REGISTERED_MODELS.items():
        for name, entry in entries.items():
            if entry["weights"] is None or hosted_here(entry["weights"]):
                continue
            kept[name] = (task, name, entry)
            former_names = [former for former, current in _DEPRECATED_MODELS[task].items() if current == name]
            former_names += [former for former, current in FORMER_KEPT_REPOSITORIES.items() if current == name]
            for former in former_names:
                kept[former] = (task, name, entry)
    return dict(sorted(kept.items()))


def rendered_config(name: str, task: Task, entry: ModelDict) -> bytes:
    """The `config.json` of a checkpoint: its registry entry without the entry point, transform, pinned URL and weight
    filter function, as the Hub repository serves it."""
    weights = entry["weights"]
    assert weights is not None
    config: Dict[str, Any] = {
        "library_name": HF_NAMESPACE,
        "name": name,
        "task": task,
        "hparams": entry["hparams"],
        "input_keys": list(entry["input_keys"]),
    }
    config.update({key: value for key, value in weights.items() if key not in ("url", "pretrained_filter_fn")})
    return (json.dumps(config, indent=2) + "\n").encode()


def card_metrics(info: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, float]]:
    """The `model-index` name and metrics of a card, `(None, {})` when the card has no `model-index`."""
    index = (info.get("cardData") or {}).get("model-index") or []
    if not index:
        return None, {}
    metrics = {metric["name"]: metric["value"] for result in index[0]["results"] for metric in result["metrics"]}
    return index[0]["name"], metrics


@pytest.fixture(scope="module")
def hub_heads() -> Dict[str, Optional[Dict[str, Any]]]:
    """The Hub API record of every hosted checkpoint's repository at its head, `None` when the repository is missing."""
    return {name: hub_api(f"models/{HF_NAMESPACE}/{name}?blobs=true") for name in hosted_checkpoints()}


def test_hosted_checkpoints_have_a_public_repository(hub_heads: Dict[str, Optional[Dict[str, Any]]]) -> None:
    """Every hosted checkpoint has a public repository in the organisation."""
    problems = [f"{name}: not on the Hub (missing or private)" for name, info in hub_heads.items() if info is None]
    problems += [f"{name}: private" for name, info in hub_heads.items() if info is not None and info.get("private")]
    assert not problems, "\n".join(problems)


def test_cards_follow_the_registry(hub_heads: Dict[str, Optional[Dict[str, Any]]]) -> None:
    """The card metadata of every hosted checkpoint names the library, the registered license and the registered
    metrics under the checkpoint's name."""
    problems: List[str] = []
    for name, (task, entry) in hosted_checkpoints().items():
        info = hub_heads[name]
        if info is None:
            continue
        weights = entry["weights"]
        assert weights is not None
        card = info.get("cardData") or {}
        if card.get("library_name") != HF_NAMESPACE:
            problems.append(f"{name}: card library_name is {card.get('library_name')!r}")
        if card.get("license") != weights["license"].lower():
            problems.append(f"{name}: card license is {card.get('license')!r}, registry says {weights['license']!r}")

        expected = {metric: float(f"{value:g}") for metric, value in (weights.get("metrics") or {}).items()}
        if not weights.get("dataset") or task == "pretraining":
            expected = {}
        indexed_name, metrics = card_metrics(info)
        if indexed_name is not None and indexed_name != name:
            problems.append(f"{name}: model-index names {indexed_name!r}")
        if metrics != expected:
            problems.append(f"{name}: card metrics {metrics}, registry metrics {expected}")
    assert not problems, "\n".join(problems)


def test_configs_follow_the_registry(hub_heads: Dict[str, Optional[Dict[str, Any]]]) -> None:
    """The `config.json` every repository serves at its head is the registry entry rendered today, in the hosted and
    the kept repositories alike."""
    problems: List[str] = []
    for name, (task, entry) in hosted_checkpoints().items():
        info = hub_heads[name]
        if info is None:
            continue
        served = repository_files(info).get("config.json")
        if served is None:
            problems.append(f"{name}: no config.json on the Hub")
        elif served["blobId"] != blob_id(rendered_config(name, task, entry)):
            problems.append(f"{name}: config.json differs from the registry entry")
    for repo, (task, name, entry) in kept_repositories().items():
        info = hub_api(f"models/{HF_NAMESPACE}/{repo}?blobs=true")
        if info is None:
            continue
        served = repository_files(info).get("config.json")
        if served is None:
            problems.append(f"{repo}: no config.json on the Hub (kept repository)")
        elif served["blobId"] != blob_id(rendered_config(name, task, entry)):
            problems.append(f"{repo}: config.json differs from the registry entry of {name!r} (kept repository)")
    assert not problems, "\n".join(problems)


def test_pinned_revisions_serve_the_head_weights(hub_heads: Dict[str, Optional[Dict[str, Any]]]) -> None:
    """The revision each registered URL pins holds the same `model.safetensors` as the repository head: the head moves
    on every card or config push, the weights must not."""
    problems: List[str] = []
    for name, (_, entry) in hosted_checkpoints().items():
        head = hub_heads[name]
        if head is None:
            continue
        weights = entry["weights"]
        assert weights is not None and isinstance(weights["url"], str)
        match = HF_URL_RE.fullmatch(weights["url"])
        assert match is not None, (name, weights["url"])
        revision = match["revision"]
        pinned = hub_api(f"models/{HF_NAMESPACE}/{name}/revision/{revision}?blobs=true")
        if pinned is None:
            problems.append(f"{name}: pinned revision {revision[:10]} does not exist")
            continue
        at_pin, at_head = repository_files(pinned), repository_files(head)
        if "model.safetensors" not in at_pin:
            problems.append(f"{name}: no model.safetensors at pinned revision {revision[:10]}")
        elif at_pin["model.safetensors"].get("lfs", {}).get("sha256") != at_head.get("model.safetensors", {}).get(
            "lfs", {}
        ).get("sha256"):
            problems.append(f"{name}: head weights differ from the pinned revision {revision[:10]} (re-pin)")
    assert not problems, "\n".join(problems)
