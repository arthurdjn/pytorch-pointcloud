"""Sync the model checkpoint catalog from the registry into a CSV.

The registry is the single source of truth for *which* checkpoints exist, whether
they ship weights, and their documented metrics (`WeightsDict.metrics`). The
script *merges* rather than overwrites: registry-derived columns are refreshed on
every run, while any metric or extra columns already filled in the CSV are
preserved per checkpoint.

The CSV backs the model catalog shown in the docs, so it always matches the
registry and the metrics you have entered.

Usage:
    uv run --no-sync python docs/scripts/build_model_tables.py
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from torch_pointcloud.models._registry import _REGISTERED_MODELS, Task

REGISTRY_COLUMNS = ["checkpoint", "architecture", "task", "dataset", "pretrained"]
DEFAULT_METRIC: Dict[Task, str] = {
    "classification": "OA",
    "segmentation": "mIoU",
    "detection": "mAP",
    "base": "",
}
TASK_ORDER: Dict[Task, int] = {"classification": 0, "segmentation": 1, "detection": 2, "base": 3}


def registry_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for task, entries in _REGISTERED_MODELS.items():
        for name, entry in entries.items():
            # Registered names follow `{arch}.{dataset}.{author}`; weightless configs may omit both tags.
            weights = entry["weights"]
            dataset = weights.get("dataset") or "" if weights else (name.split(".")[1] if "." in name else "")
            metric = DEFAULT_METRIC[task]
            score = ""
            metrics = weights.get("metrics") if weights else None
            if metrics:
                metric = metric if metric in metrics else next(iter(metrics))
                score = str(metrics[metric])
            rows.append(
                {
                    "checkpoint": name,
                    "architecture": entry["fn"].__module__.rsplit(".", 1)[-1],
                    "task": task,
                    "dataset": dataset,
                    "pretrained": str(bool(weights)).lower(),
                    "metric": metric,
                    "score": score,
                }
            )
    return rows


def sync(csv_path: Path) -> None:
    registry = pd.DataFrame(registry_rows())

    if csv_path.exists():
        existing = pd.read_csv(csv_path, dtype=str).fillna("")
        # A checkpoint name can appear under several tasks, so the key is (checkpoint, task).
        existing = existing.drop_duplicates(subset=["checkpoint", "task"])
        keep = [c for c in existing.columns if c not in REGISTRY_COLUMNS]
        merged = registry[REGISTRY_COLUMNS].merge(
            existing[["checkpoint", "task", *keep]], on=["checkpoint", "task"], how="left"
        )
        for col in keep:
            registry_default = registry[col] if col in registry else ""
            merged[col] = merged[col].where(merged[col].notna() & (merged[col] != ""), registry_default)
        registry = merged

    registry = registry.fillna("")
    registry["_order"] = registry["task"].map(TASK_ORDER).fillna(99)
    registry = registry.sort_values(["_order", "checkpoint"]).drop(columns="_order")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(csv_path, index=False)
    print(f"Synced {len(registry)} checkpoints to {csv_path}")


def main() -> None:
    docs_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sync the model catalog CSV from the registry.")
    parser.add_argument("--csv", default=docs_dir / "data" / "models.csv", type=Path)
    args = parser.parse_args()
    sync(args.csv)


if __name__ == "__main__":
    main()
