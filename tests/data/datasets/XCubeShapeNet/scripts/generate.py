"""Generate a tiny XCubeShapeNet fixture in the original release format.

The release stores per-shape pickles produced by the 2024 fvdb build XCube was developed against
(`points` is a pickled `GridBatch`, i.e. a V01-framed in-memory NanoVDB buffer, and `normals` a
`JaggedTensor` aligned to the grid's voxel order). Reproducing those bytes requires that original fvdb
(openvdb PR-1808), NOT fvdb-core: this script must run in an environment where it is installed, e.g.

    conda run -n xcube python scripts/generate.py raw ./raw

The fixture is one synthetic 194-voxel "shape" (seeded random points clustered into a few NanoVDB
leaves to keep the file small) listed in all three splits. The processed `.npz` cache is not committed:
the dataset tests exercise `XCubeShapeNet.process()` on this raw pickle directly.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import fvdb
import torch

VOXEL_SIZE = 0.01
SYNSET_ID = "03001627"
MODEL_NAME = "dummy0"


def generate_raw(raw_dir: Path) -> None:
    torch.manual_seed(0)
    points = torch.rand(200, 3) * 0.16
    grid = fvdb.sparse_grid_from_points(
        fvdb.JaggedTensor([points]), voxel_sizes=[VOXEL_SIZE] * 3, origins=[VOXEL_SIZE / 2.0] * 3
    )
    normal = torch.randn(grid.total_voxels, 3)
    normal = normal / normal.norm(dim=1, keepdim=True)

    out_dir = raw_dir / "128" / SYNSET_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"points": grid, "normals": grid.jagged_like(normal)}, out_dir / f"{MODEL_NAME}.pkl")
    for split in ("train", "val", "test"):
        (out_dir / f"{split}.lst").write_text(f"{MODEL_NAME}\n")
    print(f"wrote {grid.total_voxels}-voxel fixture -> {out_dir}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate the XCubeShapeNet test fixture.")
    parser.add_argument("command", choices=["raw"])
    parser.add_argument("raw_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_raw(args.raw_dir)


if __name__ == "__main__":
    main()
