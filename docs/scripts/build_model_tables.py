"""Sync the model checkpoint catalog from the registry into a CSV.

The registry says which checkpoints exist, whether they ship weights, their dataset and license, and those
columns are refreshed on every run. The measured columns (`reference`, what the reference implementation
publishes, and `score`, what this package measures with the reference protocol) live in the CSV and are
preserved per checkpoint. `params` (millions of parameters) is computed once per new checkpoint by
instantiating the architecture and then kept, so a run without new checkpoints is fast.

The CSV backs the model catalog in the docs and the tables of the paper.

Usage:
    uv run --no-sync python docs/scripts/build_model_tables.py
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import torch_pointcloud as tp
from torch_pointcloud.models._registry import _REGISTERED_MODELS, Task

REPO_DIR = Path(__file__).resolve().parents[2]
COLUMNS = [
    "checkpoint",
    "architecture",
    "task",
    "dataset",
    "pretrained",
    "license",
    "params",
    "metric",
    "reference",
    "score",
]
KEPT_COLUMNS = ["params", "reference", "score"]
TASK_ORDER: Dict[Task, int] = {"classification": 0, "segmentation": 1, "detection": 2, "base": 3}


DEFAULT_METRIC: Dict[Task, str] = {"classification": "OA", "segmentation": "mIoU", "detection": "mAP", "base": ""}


def metric_name(task: Task, dataset: str) -> str:
    """The catalog keeps one metric name per task; the paper refines it per dataset (instance mIoU, KITTI R11)."""
    return DEFAULT_METRIC[task]


def param_count(name: str, task: Task) -> str:
    try:
        model = tp.create_model(name, task=task, pretrained=False)
    except TypeError:
        # The config registers architecture hparams only, so its size depends on data the catalog has not got.
        return ""
    except Exception as exc:  # noqa: BLE001
        print(f"params: {name}: {type(exc).__name__}: {exc}")
        return ""
    return f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}"


def kept_values(csv_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """The columns the CSV owns, keyed by (checkpoint, task)."""
    if not csv_path.exists():
        return {}
    existing = pd.read_csv(csv_path, dtype=str).fillna("")
    columns = [c for c in KEPT_COLUMNS if c in existing.columns]
    return {(r["checkpoint"], r["task"]): {c: r[c] for c in columns} for _, r in existing.iterrows()}


def registry_rows(kept: Dict[Tuple[str, str], Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for task, entries in _REGISTERED_MODELS.items():
        for name, entry in entries.items():
            weights = entry["weights"]
            # Registered names follow `{arch}.{dataset}.{author}`; weightless configs may omit both tags.
            dataset = (weights.get("dataset") if weights else None) or (name.split(".")[1] if "." in name else "")
            previous = kept.get((name, task), {})
            rows.append(
                {
                    "checkpoint": name,
                    "architecture": entry["fn"].__module__.rsplit(".", 1)[-1],
                    "task": task,
                    "dataset": dataset,
                    "pretrained": str(bool(weights)).lower(),
                    "license": (weights.get("license") if weights else "") or "",
                    "params": previous.get("params", "") or param_count(name, task),
                    "metric": metric_name(task, dataset),
                    "reference": previous.get("reference", ""),
                    "score": previous.get("score", ""),
                }
            )
    return rows


def sync(csv_path: Path) -> None:
    table = pd.DataFrame(registry_rows(kept_values(csv_path)), columns=COLUMNS)
    table["_order"] = table["task"].map(TASK_ORDER).fillna(99)
    table = table.sort_values(["_order", "checkpoint"]).drop(columns="_order")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    pretrained = table[table["pretrained"] == "true"]
    unmeasured = sorted(pretrained[pretrained["score"] == ""]["checkpoint"])
    print(
        f"Synced {len(table)} checkpoints to {csv_path}; {len(pretrained) - len(unmeasured)}/{len(pretrained)} pretrained ones measured"
    )
    if unmeasured:
        print("no measured score yet:", ", ".join(unmeasured))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the model catalog CSV from the registry.")
    parser.add_argument("--csv", default=REPO_DIR / "docs" / "data" / "models.csv", type=Path)
    args = parser.parse_args()
    sync(args.csv)


if __name__ == "__main__":
    main()
