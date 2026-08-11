"""Generate ShapeNet shapes with the two-stage XCube pipeline.

Samples coarse $128^3$ structures with the dense latent diffusion model, then upsamples each one to
$512^3$ with the sparse diffusion model conditioned on the coarse normals. Voxel centers and normals of
both stages are saved as one `.npz` per shape.

Usage:
    uv run --no-sync python examples/xcube_generate_shapenet.py --category chair --num-samples 4
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

from torch_pointcloud.models import create_model
from torch_pointcloud.models.xcube import XCubeDiffusion
from torch_pointcloud.utils.random import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    coarse = create_model(f"xcube-diffusion-coarse-nvidia.shapenet-{args.category}", task="base", pretrained=True)
    fine = create_model(f"xcube-diffusion-fine-nvidia.shapenet-{args.category}", task="base", pretrained=True)
    assert isinstance(coarse, XCubeDiffusion) and isinstance(fine, XCubeDiffusion)
    coarse.to(args.device).eval()
    fine.to(args.device).eval()

    generated = 0
    while generated < args.num_samples:
        batch = min(args.batch_size, args.num_samples - generated)
        out = coarse.sample(batch_size=batch, num_steps=args.num_steps)
        out = fine.sample(grid=out["grid"], normal=out["normal"], num_steps=args.num_steps)
        grid, normal = out["grid"], out["normal"]
        pos = grid.voxel_to_world(grid.ijk.float()).jdata
        jidx = grid.ijk.jidx.long()
        for b in range(batch):
            mask = jidx == b
            path = out_dir / f"{args.category}_{generated:04d}.npz"
            np.savez(path, pos=pos[mask].cpu().numpy(), normal=normal[mask].cpu().numpy())
            print(f"{path}: {int(mask.sum())} voxels")
            generated += 1


def parse_args() -> Namespace:
    parser = ArgumentParser(description="XCube two-stage ShapeNet generation.")
    parser.add_argument("--category", type=str, default="chair", choices=["chair", "car", "plane"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--out", type=str, default="results/xcube")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
