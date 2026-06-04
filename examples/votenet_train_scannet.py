r"""Minimal VoteNet training example: model + `VoteNetLoss` + optimizer, end to end.

This is a self-contained smoke-test of the VoteNet training pipeline on ScanNet
detection data. It wires the registered `votenet-fair-base.scannet` model to the
`VoteNetLoss` objective and runs a short Adam loop over a tiny fixed batch, printing
the total loss and its sub-losses at every step. Overfitting a few real scenes drives
the mean total loss down by roughly a third over 30 steps (with all sub-losses
trending down), which confirms the gradients flow correctly through the packed model
into the dense loss. The per-step loss is noisy because farthest-point sampling draws
a fresh random start each forward in `train()` mode, so the verdict compares the mean
of the first window of steps against the last.

Data:
    If a directory of preprocessed ScanNet detection scenes is available it is used,
    otherwise a synthetic batch (random points with a few axis-aligned boxes) is
    generated so the example always runs. The synthetic fallback is only a smoke-test
    that the pipeline executes; its random boxes are a weak signal that does not
    reliably reduce the loss, so use real data to see the decrease. Point it at real
    data with `--data-root`: a directory holding per-scene `{scene}_vert.npy`
    (xyz, optionally + rgb) and `{scene}_bbox.npy` of shape $(K, 7)$ =
    $[c_x, c_y, c_z, d_x, d_y, d_z, \text{nyu40}]$, as produced by the
    facebookresearch/votenet ScanNet preprocessing. The default `--data-root` is `/tmp/scannet_det`.

Scope:
    This trains from random initialization on a tiny batch purely to demonstrate the
    pipeline. Full-scale training (the real dataset, augmentation, an eval loop and a
    Lightning harness) is a follow-up; this script only proves the loss decreases.

Usage:
    uv run --no-sync python examples/votenet_train_scannet.py
    uv run --no-sync python examples/votenet_train_scannet.py --data-root /path/to/scannet_det
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from torch import Tensor

from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.models import VoteNetDetection, create_model

NYU40_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
MAX_NUM_OBJ = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_scenes(data_root: Path, limit: int) -> List[Dict[str, np.ndarray]]:
    r"""Load up to `limit` preprocessed ScanNet detection scenes, or an empty list if none exist.

    Args:
        data_root: Directory holding per-scene `{scene}_vert.npy` and `{scene}_bbox.npy` files.
        limit: Maximum number of scenes to return.

    Returns:
        A list of `{"vert": (N, 3+), "bbox": (K, 7)}` dicts (empty if `data_root` has no scenes).
    """
    if not data_root.is_dir():
        return []
    verts = sorted(data_root.glob("*_vert.npy"))
    scenes: List[Dict[str, np.ndarray]] = []
    for vert_path in verts[:limit]:
        bbox_path = data_root / vert_path.name.replace("_vert.npy", "_bbox.npy")
        if not bbox_path.exists():
            continue
        bbox = np.load(bbox_path)
        if bbox.ndim == 1:
            bbox = bbox.reshape(0, 7)
        scenes.append({"vert": np.load(vert_path), "bbox": bbox})
    return scenes


def synthesize_scenes(num_scenes: int, num_points: int, mean_size_arr: np.ndarray) -> List[Dict[str, np.ndarray]]:
    r"""Build random scenes with a few axis-aligned boxes when no real data is available.

    Each scene scatters `num_points` points in a $4\,\text{m}$ cube and places a few boxes whose
    centers, sizes (mean template + jitter) and classes are random. This fallback only exists so the
    example runs anywhere; random boxes give a weak, noisy training signal, so point the script at real
    ScanNet detection data to see the loss fall cleanly. The point count stays above the registered
    transform's sample size so the transform fixes a uniform $N$.

    Args:
        num_scenes: Number of scenes to generate.
        num_points: Points per scene.
        mean_size_arr: Per-class mean box sizes, shape $(\text{num\_size\_cluster}, 3)$.

    Returns:
        A list of `{"vert": (N, 3), "bbox": (K, 7)}` dicts in the ScanNet detection layout.
    """
    rng = np.random.default_rng(0)
    num_class = mean_size_arr.shape[0]
    scenes: List[Dict[str, np.ndarray]] = []
    for _ in range(num_scenes):
        vert = rng.uniform(0.0, 4.0, size=(num_points, 3)).astype("float32")
        num_obj = int(rng.integers(3, 6))
        classes = rng.integers(0, num_class, size=num_obj)
        centers = rng.uniform(0.5, 3.5, size=(num_obj, 3))
        sizes = mean_size_arr[classes] * rng.uniform(0.8, 1.2, size=(num_obj, 3))
        nyu = NYU40_IDS[classes]
        bbox = np.concatenate([centers, sizes, nyu[:, None]], axis=1).astype("float32")
        scenes.append({"vert": vert, "bbox": bbox})
    return scenes


def encode_scene(pos: Tensor, bbox: np.ndarray, mean_size_arr: Tensor) -> Dict[str, Tensor]:
    r"""Encode one scene's GT boxes and per-point votes to the dense tensors the loss expects.

    ScanNet boxes are axis-aligned, so every heading label is $0$. The NYU40 semantic id of
    each box is mapped to a contiguous $0\ldots\text{num\_class}-1$ class. A point is a vote
    source when it falls inside any box (axis-aligned containment on the sampled `pos`); its
    vote target is the offset to that box's center, tiled three times to fill the $9$-vector.

    Args:
        pos: Sampled scene points, shape $(N, 3)$.
        bbox: GT boxes, shape $(K, 7)$ = $[c_x, c_y, c_z, d_x, d_y, d_z, \text{nyu40}]$.
        mean_size_arr: Per-class mean box sizes, shape $(\text{num\_size\_cluster}, 3)$.

    Returns:
        A dict of per-scene GT tensors: dense object labels padded to `MAX_NUM_OBJ`
        (`center_label`, `heading_class_label`, `heading_residual_label`, `size_class_label`,
        `size_residual_label`, `sem_cls_label`, `box_label_mask`) and per-point vote targets
        (`vote_label` $(N, 9)$, `vote_label_mask` $(N,)$).
    """
    device = pos.device
    n = pos.size(0)
    nyu2class = {int(nyu): i for i, nyu in enumerate(NYU40_IDS)}

    center_label = torch.zeros(MAX_NUM_OBJ, 3, device=device)
    size_class_label = torch.zeros(MAX_NUM_OBJ, dtype=torch.long, device=device)
    size_residual_label = torch.zeros(MAX_NUM_OBJ, 3, device=device)
    sem_cls_label = torch.zeros(MAX_NUM_OBJ, dtype=torch.long, device=device)
    box_label_mask = torch.zeros(MAX_NUM_OBJ, device=device)

    vote_label = torch.zeros(n, 9, device=device)
    vote_label_mask = torch.zeros(n, device=device)

    boxes = torch.from_numpy(bbox).to(device=device, dtype=torch.float32)
    mean_size = mean_size_arr.to(device)
    obj = 0
    for i in range(boxes.size(0)):
        nyu = int(boxes[i, 6].item())
        if nyu not in nyu2class or obj >= MAX_NUM_OBJ:
            # The 18-class detector ignores boxes outside its label set (e.g. wall) and any past the cap.
            continue
        center = boxes[i, 0:3]
        size = boxes[i, 3:6]
        cls = nyu2class[nyu]

        center_label[obj] = center
        size_class_label[obj] = cls
        size_residual_label[obj] = size - mean_size[cls]
        sem_cls_label[obj] = cls
        box_label_mask[obj] = 1.0
        obj += 1

        half = size / 2.0
        inside = ((pos >= center - half) & (pos <= center + half)).all(dim=1)
        offset = center - pos[inside]
        vote_label[inside] = offset.repeat(1, 3)
        vote_label_mask[inside] = 1.0

    return {
        "center_label": center_label,
        "heading_class_label": torch.zeros(MAX_NUM_OBJ, dtype=torch.long, device=device),
        "heading_residual_label": torch.zeros(MAX_NUM_OBJ, device=device),
        "size_class_label": size_class_label,
        "size_residual_label": size_residual_label,
        "sem_cls_label": sem_cls_label,
        "box_label_mask": box_label_mask,
        "vote_label": vote_label,
        "vote_label_mask": vote_label_mask,
    }


def build_batch(
    scenes: List[Dict[str, np.ndarray]],
    transform: Callable[[Dict[str, Any]], Dict[str, Any]],
    mean_size_arr: Tensor,
) -> Dict[str, Tensor]:
    r"""Transform each scene, pack the inputs and stack the dense GT into a single batch.

    Args:
        scenes: Per-scene `{"vert", "bbox"}` dicts.
        transform: The model's registered transform (adds the height feature and samples a fixed $N$).
        mean_size_arr: Per-class mean box sizes, shape $(\text{num\_size\_cluster}, 3)$.

    Returns:
        A dict with the packed model inputs (`pos`, `x`, `batch`) and the dense per-scene GT
        tensors required by `VoteNetLoss`, with batch as the leading dimension. All scenes must
        share the same sampled $N$ so the per-point vote labels stack cleanly.
    """
    pos_list: List[Tensor] = []
    x_list: List[Tensor] = []
    batch_list: List[Tensor] = []
    gt_list: List[Dict[str, Tensor]] = []

    for i, scene in enumerate(scenes):
        pos = torch.from_numpy(scene["vert"][:, 0:3].astype("float32"))
        sample = transform({"pos": pos})
        pos_s = sample["pos"].to(DEVICE)
        x_s = sample["x"].to(DEVICE)
        pos_list.append(pos_s)
        x_list.append(x_s)
        batch_list.append(torch.full((pos_s.size(0),), i, dtype=torch.long, device=DEVICE))
        gt_list.append(encode_scene(pos_s, scene["bbox"], mean_size_arr))

    sizes = {p.size(0) for p in pos_list}
    if len(sizes) != 1:
        raise ValueError(f"All scenes must share the same sampled point count, got {sorted(sizes)}.")

    batch: Dict[str, Tensor] = {
        "pos": torch.cat(pos_list, dim=0),
        "x": torch.cat(x_list, dim=0),
        "batch": torch.cat(batch_list, dim=0),
    }
    for key in gt_list[0]:
        batch[key] = torch.stack([g[key] for g in gt_list], dim=0)
    return batch


def densify_seeds(output: Dict[str, Tensor], batch_size: int, num_points: int) -> Dict[str, Tensor]:
    r"""Reshape the model's packed seed / vote tensors to the dense layout the loss expects.

    The seeds are a fixed, contiguous-per-scene count, so the packed $(\sum_i S_i, \cdot)$ tensors
    reshape to $(B, S, \cdot)$ with $S = \sum_i S_i // B$ without any gather. The model's `seed_inds`
    index into the concatenated packed input, so the per-scene offset $i \cdot N$ is subtracted to make
    them scene-local indices into the dense $(B, N, \cdot)$ vote labels.

    Args:
        output: The packed `VoteNetOutput` from the model.
        batch_size: Number of scenes $B$.
        num_points: Points per scene $N$ (uniform after the fixed-size sampling transform).

    Returns:
        A copy of `output` with `seed_xyz`, `vote_xyz` as $(B, S, 3)$ and scene-local `seed_inds` as $(B, S)$.
    """
    seed_xyz = output["seed_xyz"]
    num_seed = seed_xyz.size(0) // batch_size
    seed_inds = output["seed_inds"].view(batch_size, num_seed)
    offset = torch.arange(batch_size, device=seed_inds.device).view(batch_size, 1) * num_points
    dense = dict(output)
    dense["seed_xyz"] = seed_xyz.view(batch_size, num_seed, 3)
    dense["vote_xyz"] = output["vote_xyz"].view(batch_size, num_seed, 3)
    dense["seed_inds"] = seed_inds - offset
    return dense


def train(args: Namespace) -> None:
    r"""Run the short overfitting loop and print the per-step losses.

    Args:
        args: Parsed command-line arguments (`data_root`, `num_scenes`, `num_points`, `steps`, `lr`).
    """
    model, info = create_model("votenet-fair-base.scannet", task="detection", return_info=True)
    assert isinstance(model, VoteNetDetection)
    model = model.to(DEVICE).train()
    transform = info["transforms"]
    mean_size_arr_np = model.mean_size_arr.cpu().numpy()

    criterion = VoteNetLoss(
        num_heading_bin=model.num_heading_bin,
        num_size_cluster=model.num_size_cluster,
        num_class=model.num_classes,
        mean_size_arr=model.mean_size_arr,
    ).to(DEVICE)

    scenes = load_scenes(Path(args.data_root), args.num_scenes)
    if scenes:
        print(f"Loaded {len(scenes)} real ScanNet scene(s) from {args.data_root}.")
    else:
        scenes = synthesize_scenes(args.num_scenes, args.num_points, mean_size_arr_np)
        print(f"No data under {args.data_root}; using {len(scenes)} synthetic scene(s).")

    batch = build_batch(scenes, transform, model.mean_size_arr)
    batch_size = len(scenes)
    num_points = batch["pos"].size(0) // batch_size
    print(f"Batch: {batch_size} scenes, {num_points} points each, device={DEVICE}.\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: List[float] = []
    for step in range(args.steps):
        optimizer.zero_grad()
        output = model(batch["x"], batch["pos"], batch["batch"])
        losses = criterion(densify_seeds(output, batch_size, num_points), batch)
        losses["loss"].backward()
        optimizer.step()

        history.append(losses["loss"].item())
        print(
            f"step {step:3d} | loss {history[-1]:7.3f} | vote {losses['vote_loss'].item():6.3f} "
            f"| obj {losses['objectness_loss'].item():6.3f} | box {losses['box_loss'].item():6.3f} "
            f"| sem {losses['sem_cls_loss'].item():6.3f} | obj_acc {losses['obj_acc'].item():.3f}"
        )

    # FPS random starts and BatchNorm make the per-step loss noisy, so compare windowed means.
    window = max(1, len(history) // 5)
    start = sum(history[:window]) / window
    end = sum(history[-window:]) / window
    print(f"\nMean total loss: {start:.3f} (first {window} steps) -> {end:.3f} (last {window} steps).")
    if end < start:
        print(f"Loss decreased by {start - end:.3f} ({100 * (1 - end / start):.1f}%).")
    else:
        print("Loss did not decrease; check the data or learning rate.")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Minimal VoteNet training-pipeline example on ScanNet detection data.")
    parser.add_argument("--data-root", type=str, default="/tmp/scannet_det", help="Dir with *_vert.npy / *_bbox.npy.")
    parser.add_argument("--num-scenes", type=int, default=2, help="Scenes in the overfitting batch.")
    parser.add_argument("--num-points", type=int, default=50000, help="Points per synthetic scene (before sampling).")
    parser.add_argument("--steps", type=int, default=30, help="Optimizer steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
