"""Generate per-module stub .md files under `docs/api/`.

Each stub is a one-liner that hands rendering off to mkdocstrings:

```markdown
::: torch_pointcloud.transforms.transforms
```

The actual summary tables (classes / functions per module) are produced by
mkdocstrings' built-in `summary` option, configured in `zensical.toml`.
This script's only job is to ensure that some .md file exists for every
public submodule, so the nav and the API tree match the source layout.

Usage:
    uv run --no-sync python docs/scripts/build_api_reference.py torch_pointcloud --out docs/api
"""

import argparse
import shutil
import sys
from pathlib import Path

import griffe
from griffe import Module

DISPLAY_TITLES = {
    "pointnet": "PointNet",
    "pointnet2": "PointNet++",
    "dgcnn": "DGCNN",
    "pointcnn": "PointCNN",
    "pointconv": "PointConv",
    "pointmlp": "PointMLP",
    "pointnext": "PointNeXt",
    "point_transformer": "Point Transformer",
    "point_transformer_v2": "Point Transformer V2",
    "point_transformer_v3": "Point Transformer V3",
    "pvcnn": "PVCNN",
    "pvcnn2": "PVCNN++",
    "point_mamba": "Point-Mamba",
    "octformer": "OctFormer",
    "kpconv": "KPConv",
    "randlanet": "RandLA-Net",
    "spvcnn": "SPVCNN",
    "spunet": "SpUNet",
    "spformer_unet": "SPFormer-UNet",
    "sphereformer": "SphereFormer",
    "oneformer3d": "OneFormer3D",
    "votenet": "VoteNet",
    "detr3d": "3DETR",
    "pointpillars": "PointPillars",
    "second": "SECOND",
    "pointrcnn": "PointRCNN",
    "voxelnext": "VoxelNeXt",
    "voxel_mamba": "Voxel-Mamba",
    "lion": "LION",
    "point_bert": "Point-BERT",
    "point_mae": "Point-MAE",
    "point_m2ae": "Point-M2AE",
    "pointgpt": "PointGPT",
    "sonata": "Sonata",
    "concerto": "Concerto",
    "utonia": "Utonia",
    "modelnet": "ModelNet",
    "scannet": "ScanNet",
    "s3dis": "S3DIS",
    "shapenetpart": "ShapeNetPart",
    "scanobjectnn": "ScanObjectNN",
    "semantickitti": "SemanticKITTI",
    "semantic3d": "Semantic3D",
    "toronto3d": "Toronto3D",
    "parislille3d": "ParisLille3D",
    "sunrgbd": "SUN RGB-D",
    "kitti": "KITTI",
    "nuscenes": "nuScenes",
}


# `DictTransform.iter_keys` is inherited unchanged by every transform; render it once, on its own class.
MODULE_OPTIONS = {
    "torch_pointcloud.transforms": {"inherited_members": "false"},
    "torch_pointcloud.transforms.transforms": {"inherited_members": "false"},
}


def has_public_submodules(module: Module) -> bool:
    return any(isinstance(m, Module) and not m.is_private and not m.is_alias for m in module.members.values())


def write_stub(module: Module, output_dir: Path) -> None:
    """Write a one-line `::: module.path` stub at the right location.

    The stub carries an explicit anchor for the module. The rendered pages hide the root
    heading, which is what would otherwise register the module identifier, so without the
    anchor a cross-reference to the module resolves to plain text.
    """
    rel = "/".join(module.path.split(".")[1:])
    submodules = has_public_submodules(module)
    if rel == "":
        file_path = output_dir / "index.md"
    elif submodules:
        file_path = output_dir / rel / "index.md"
    else:
        file_path = output_dir / f"{rel}.md"

    file_path.parent.mkdir(parents=True, exist_ok=True)
    name = module.path.split(".")[-1]
    title = DISPLAY_TITLES.get(name, name)
    directive = f"::: {module.path}\n"
    options = MODULE_OPTIONS.get(module.path)
    if options:
        directive += "    options:\n" + "".join(f"      {key}: {value}\n" for key, value in options.items())

    anchor = f"[](){{#{module.path}}}\n"
    file_path.write_text(f"---\ntitle: {title}\n---\n\n{anchor}\n{directive}")
    print(f"Generated: {file_path}")


def process_module(module: Module, output_dir: Path) -> None:
    write_stub(module, output_dir)
    for child in module.members.values():
        if isinstance(child, Module) and not child.is_private and not child.is_alias:
            process_module(child, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stub .md files for mkdocstrings.")
    parser.add_argument("package", default="torch_pointcloud", help="Name of the package to load")
    parser.add_argument("--out", "-o", default="docs/api", help="Output directory")
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        package = griffe.load(args.package)
    except ImportError as exc:
        print(f"Error loading package: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating stubs in: {output_path}")
    process_module(package, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
