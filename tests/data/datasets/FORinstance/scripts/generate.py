"""Generate a tiny synthetic FOR-instance fixture in the layout of the extracted release.

The release extracts to `<collection>/<plot>.las` next to `data_split_metadata.csv`: one LAS 1.2 file per plot in UTM
coordinates, the class code in the classification field and the tree id in a `treeID` extra dimension (int32, or
float64 for RMIT and TUWIEN). No plot of the release is read: every point is drawn from a seeded generator, trees as
a stem under a cone of live branches over a flat terrain with low vegetation, and one tree outside the plot.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import laspy
import numpy as np

# Plots of the fixture, their split and the dtype of their `treeID` dimension.
PLOTS = {
    "CULS/plot_1_annotated.las": ("dev", np.int32),
    "CULS/plot_2_annotated.las": ("test", np.int32),
    "TUWIEN/test.las": ("test", np.float64),
}
# A plot of the split file that the archive does not hold, as the release's NIBIO2 rows.
MISSING = "NIBIO2/plot1_annotated.las"
PLOT_SIZE = 20.0
NUM_TREES = 4


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic FOR-instance test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument("--points-per-tree", type=int, default=600, help="Points per tree.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def tree_points(rng: np.random.Generator, center: np.ndarray, height: float, num_points: int) -> np.ndarray:
    """Points of one tree: a stem up to a third of its height, then live branches in a cone, as (x, y, z, class)."""
    num_stem = num_points // 5
    z_stem = rng.uniform(0.0, height / 3, num_stem)
    stem = np.stack([center[0] + rng.normal(0, 0.1, num_stem), center[1] + rng.normal(0, 0.1, num_stem), z_stem], 1)
    num_crown = num_points - num_stem
    z_crown = rng.uniform(height / 3, height, num_crown)
    radius = (height - z_crown) / 3 * np.sqrt(rng.uniform(0, 1, num_crown))
    angle = rng.uniform(0, 2 * np.pi, num_crown)
    crown = np.stack([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), z_crown], 1)
    classes = np.concatenate([np.full(num_stem, 4), rng.choice([5, 5, 5, 6], num_crown)])
    return np.concatenate([np.concatenate([stem, crown]), classes[:, None]], axis=1)


def generate_plot(rng: np.random.Generator, points_per_tree: int, dtype: type) -> laspy.LasData:
    parts, ids = [], []
    for tree in range(NUM_TREES):
        center = np.array([5.0 + 10.0 * (tree % 2), 5.0 + 10.0 * (tree // 2)]) + rng.normal(0, 0.5, 2)
        parts.append(tree_points(rng, center, rng.uniform(8.0, 14.0), points_per_tree))
        ids.append(np.full(points_per_tree, 1101 + tree))
    outside = tree_points(rng, np.array([PLOT_SIZE + 3.0, PLOT_SIZE / 2]), 10.0, points_per_tree)
    outside[:, 3] = 3
    parts.append(outside)
    ids.append(np.full(points_per_tree, 9001))
    num_ground = 4 * points_per_tree
    ground = np.stack(
        [
            rng.uniform(0.0, PLOT_SIZE, num_ground),
            rng.uniform(0.0, PLOT_SIZE, num_ground),
            rng.uniform(-0.05, 0.05, num_ground),
            rng.choice([1, 2, 2, 2], num_ground),
        ],
        axis=1,
    )
    ground[ground[:, 3] == 1, 2] += rng.uniform(0.1, 1.0, int((ground[:, 3] == 1).sum()))
    parts.append(ground)
    ids.append(np.zeros(num_ground))
    points, tree_id = np.concatenate(parts), np.concatenate(ids)

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = np.array([600000.0, 6600000.0, 100.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=dtype))
    las = laspy.LasData(header)
    las.x = 600000.0 + points[:, 0]
    las.y = 6600000.0 + points[:, 1]
    las.z = 100.0 + points[:, 2]
    las.classification = points[:, 3].astype(np.uint8)
    las["treeID"] = tree_id.astype(dtype)
    return las


def generate_raw(args: Namespace) -> None:
    dst_root = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)

    rows = ["path,folder,split"]
    for path, (split, dtype) in PLOTS.items():
        dst = dst_root / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        generate_plot(rng, args.points_per_tree, dtype).write(dst.as_posix())
        rows.append(f"{path},{Path(path).parent.name},{split}")
        print(f"  {dst}")
    rows.append(f"{MISSING},{Path(MISSING).parent.name},test")
    (dst_root / "data_split_metadata.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
