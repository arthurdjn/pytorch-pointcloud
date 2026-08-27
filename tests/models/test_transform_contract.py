"""Every registered transform leaves the sample in the one sampling-key contract.

`pos`, `x`, `segment` share the predictor resolution; a pipeline that changes the number of points keeps the
source cloud under `origin_pos` (and `origin_segment` when it carries labels) and records exactly one row
map: `inverse` (source row to predictor row, voxelizers) or `index` (predictor row to source row, selection
samplers). No other per-point tensor may be left at source resolution, and the output must collate.
"""

from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch

from torch_pointcloud.models._registry import _REGISTERED_MODELS
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)

OBJECT_POINTS = 8704  # above the largest registered FPS budget (8192)
SCENE_POINTS = 4000  # plus a quarter duplicated so every voxel size merges rows


def _scene_pos(g: torch.Generator, scale: torch.Tensor) -> torch.Tensor:
    pos = torch.rand(SCENE_POINTS, 3, generator=g) * scale
    return torch.cat([pos, pos[: SCENE_POINTS // 4]])


def _unit(g: torch.Generator, n: int) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)


def modelnet(g: torch.Generator) -> Dict[str, Any]:
    return {
        DataKeys.POS: torch.randn(OBJECT_POINTS, 3, generator=g),
        DataKeys.NORMAL: _unit(g, OBJECT_POINTS),
        DataKeys.LABEL: torch.tensor(3),
    }


def scanobjectnn(g: torch.Generator) -> Dict[str, Any]:
    return {
        DataKeys.POS: torch.randn(OBJECT_POINTS, 3, generator=g),
        DataKeys.LABEL: torch.tensor(3),
    }


def shapenetpart(g: torch.Generator) -> Dict[str, Any]:
    return {
        DataKeys.POS: torch.randn(OBJECT_POINTS, 3, generator=g),
        DataKeys.NORMAL: _unit(g, OBJECT_POINTS),
        DataKeys.SEGMENT: torch.randint(0, 50, (OBJECT_POINTS,), generator=g),
        DataKeys.CATEGORY: torch.tensor(4),
    }


def s3dis(g: torch.Generator) -> Dict[str, Any]:
    pos = _scene_pos(g, torch.tensor([6.0, 6.0, 3.0]))
    n = pos.shape[0]
    return {
        DataKeys.POS: pos,
        DataKeys.COLOR: torch.randint(0, 256, (n, 3), generator=g, dtype=torch.uint8),
        DataKeys.SEGMENT: torch.randint(0, 13, (n,), generator=g),
        DataKeys.INSTANCE: torch.randint(0, 20, (n,), generator=g),
    }


def s3dis_blocks(g: torch.Generator) -> Dict[str, Any]:
    pos = _scene_pos(g, torch.tensor([1.0, 1.0, 3.0]))
    n = pos.shape[0]
    return {
        DataKeys.POS: pos,
        DataKeys.COLOR: torch.rand(n, 3, generator=g),
        DataKeys.NORM_POS: torch.rand(n, 3, generator=g),
        DataKeys.SEGMENT: torch.randint(0, 13, (n,), generator=g),
    }


def scannet(g: torch.Generator) -> Dict[str, Any]:
    pos = _scene_pos(g, torch.tensor([6.0, 6.0, 3.0]))
    n = pos.shape[0]
    return {
        DataKeys.POS: pos,
        DataKeys.COLOR: torch.randint(0, 256, (n, 3), generator=g, dtype=torch.uint8),
        DataKeys.NORMAL: _unit(g, n),
        DataKeys.SEGMENT: torch.randint(0, 41, (n,), generator=g),
        DataKeys.INSTANCE: torch.randint(0, 20, (n,), generator=g),
        DataKeys.SCENE: "scene0000_00",
    }


def semantickitti(g: torch.Generator) -> Dict[str, Any]:
    pos = _scene_pos(g, torch.tensor([100.0, 100.0, 8.0])) - torch.tensor([50.0, 50.0, 4.0])
    n = pos.shape[0]
    return {
        DataKeys.POS: pos,
        DataKeys.INTENSITY: torch.rand(n, 1, generator=g),
        DataKeys.SEGMENT: torch.tensor([0, 10, 40, 50])[torch.randint(0, 4, (n,), generator=g)],
        DataKeys.INSTANCE: torch.randint(0, 20, (n,), generator=g),
        DataKeys.SEQUENCE: "00",
        DataKeys.FRAME: "000000",
    }


def nuscenes_lidarseg(g: torch.Generator) -> Dict[str, Any]:
    pos = _scene_pos(g, torch.tensor([100.0, 100.0, 8.0])) - torch.tensor([50.0, 50.0, 4.0])
    n = pos.shape[0]
    return {
        DataKeys.POS: pos,
        DataKeys.INTENSITY: torch.rand(n, 1, generator=g),
        DataKeys.SEGMENT: torch.randint(0, 32, (n,), generator=g),
    }


def sunrgbd(g: torch.Generator) -> Dict[str, Any]:
    pos = _scene_pos(g, torch.tensor([6.0, 6.0, 3.0]))
    n = pos.shape[0]
    return {
        DataKeys.POS: pos,
        DataKeys.COLOR: torch.rand(n, 3, generator=g),
        DataKeys.BOX: torch.rand(3, 7, generator=g),
        DataKeys.LABEL: torch.randint(0, 10, (3,), generator=g),
    }


def kitti(g: torch.Generator) -> Dict[str, Any]:
    pos = _scene_pos(g, torch.tensor([70.0, 80.0, 4.0])) + torch.tensor([0.0, -40.0, -3.0])
    n = pos.shape[0]
    return {
        DataKeys.POS: pos,
        DataKeys.INTENSITY: torch.rand(n, 1, generator=g),
        DataKeys.BOX: torch.rand(3, 7, generator=g),
        DataKeys.LABEL: torch.randint(0, 3, (3,), generator=g),
        DataKeys.TRUNCATION: torch.rand(3, generator=g),
        DataKeys.OCCLUSION: torch.randint(0, 3, (3,), generator=g),
        DataKeys.BBOX_HEIGHT: torch.rand(3, generator=g) * 50,
        DataKeys.FRAME: "000000",
    }


BUILDERS: Dict[str, Callable[[torch.Generator], Dict[str, Any]]] = {
    "modelnet": modelnet,
    "scanobjectnn": scanobjectnn,
    "shapenetpart": shapenetpart,
    "s3dis": s3dis,
    "s3dis_blocks": s3dis_blocks,
    "scannet": scannet,
    "semantickitti": semantickitti,
    "nuscenes_lidarseg": nuscenes_lidarseg,
    "sunrgbd": sunrgbd,
    "kitti": kitti,
}


def _cases(task: str, family: str, names: List[str]) -> List[Any]:
    return [pytest.param(task, name, family, id=name) for name in names]


CASES: List[Any] = [
    *_cases(
        "base",
        "scannet",
        [
            "concerto-tiny.pretrain.pointcept",
            "concerto-small.pretrain.pointcept",
            "concerto-base.pretrain.pointcept",
            "concerto-large.pretrain.pointcept",
            "sonata-base.pretrain.fair",
            "spformer-unet.scannet",
            "utonia.pretrain.pointcept",
        ],
    ),
    *_cases(
        "classification",
        "modelnet",
        [
            "dgcnn.modelnet40-1024.an-tao",
            "dgcnn.modelnet40-2048.an-tao",
            "point-bert-base.modelnet40.xumin-yu",
            "point-bert-base.modelnet40-4k.xumin-yu",
            "point-bert-base.modelnet40-8k.xumin-yu",
            "point-m2ae-base.modelnet40.renrui-zhang",
            "point-mae-base.modelnet40.yatian-pang",
            "point-mae-base.modelnet40-8k.yatian-pang",
            "point-mamba-base.modelnet40.dingkang-liang",
            "point-transformer.modelnet40",
            "pointconv-density-base.modelnet40.wenxuan-wu",
            *[f"pointgpt-{s}.modelnet40.guangyan-chen" for s in ("s", "b", "l")],
            *[f"pointgpt-{s}.modelnet40-8k.guangyan-chen" for s in ("s", "b", "l")],
            "pointmlp-base.modelnet40.xu-ma",
            "pointmlp-elite.modelnet40.xu-ma",
            "pointnet.modelnet40",
            "pointnet2-msg.modelnet40.xu-yan",
            "pointnet2-ssg.modelnet40.xu-yan",
            "pointnet2.modelnet40.openpoints",
            "pointnext-sm-c64.modelnet40.openpoints",
        ],
    ),
    *_cases(
        "classification",
        "scanobjectnn",
        [
            "point-bert-base.scanobjectnn-hardest.xumin-yu",
            "point-bert-base.scanobjectnn-objbg.xumin-yu",
            "point-bert-base.scanobjectnn-objonly.xumin-yu",
            "point-m2ae-base.scanobjectnn-hardest.renrui-zhang",
            "point-m2ae-base.scanobjectnn-objbg.renrui-zhang",
            "point-mamba-base.scanobjectnn.dingkang-liang",
            "point-mamba-base.scanobjectnn-nobg.dingkang-liang",
            "point-mamba-base.scanobjectnn-augmentedrot-scale75.dingkang-liang",
            *[
                f"pointgpt-{s}.scanobjectnn-{v}.guangyan-chen"
                for s in ("s", "b", "l")
                for v in ("hardest", "objbg", "objonly")
            ],
            "pointmlp-base.scanobjectnn.xu-ma",
            "pointmlp-elite.scanobjectnn.xu-ma",
            "pointnet2.scanobjectnn.openpoints",
            "pointnext-sm.scanobjectnn.openpoints",
        ],
    ),
    *_cases("detection", "scannet", ["3detr.scannet.fair", "3detr-m.scannet.fair", "votenet.scannet.fair"]),
    *_cases("detection", "sunrgbd", ["3detr.sunrgbd.fair", "votenet.sunrgbd.fair"]),
    *_cases("detection", "kitti", ["pointrcnn.kitti.openpcdet"]),
    *_cases(
        "segmentation",
        "scannet",
        [
            "concerto-large-lp.scannet20.pointcept",
            "oneformer3d-base.scannet20.danila-rukhovich",
            "oneformer3d-base.scannet200.danila-rukhovich",
            "point-transformer.scannet20",
            "ptv2-base.scannet20",
            "ptv2-base.scannet200",
            "ptv3-base.scannet20.pointcept",
            "ptv3-base.scannet200.pointcept",
            "sonata-lp.scannet20.fair",
            "spformer-unet.scannet20",
            "spunet-v1m1.scannet20.pointcept",
            "utonia-lp.scannet20.pointcept",
        ],
    ),
    *_cases(
        "segmentation",
        "s3dis",
        [
            "oneformer3d-base.s3dis-area5.danila-rukhovich",
            "point-transformer.s3dis-area5",
            *[f"pointnet2.s3dis-area{i}.openpoints" for i in range(1, 7)],
            "ptv3-base.s3dis-area5.pointcept",
        ],
    ),
    *_cases(
        "segmentation",
        "s3dis_blocks",
        [
            *[f"dgcnn.s3dis-area{i}.an-tao" for i in range(1, 7)],
            "kpfcnn-base.s3dis.hugues-thomas",
            "kpfcnn-base-sm.s3dis.hugues-thomas",
            "kpfcnn-base-deform.s3dis.hugues-thomas",
            "kpfcnn-base-sm-deform.s3dis.hugues-thomas",
            "pointnet.s3dis-area5",
            "pointnet2.s3dis-area5.xu-yan",
            *[f"pointnext-{s}.s3dis-area{i}.openpoints" for s in ("sm", "base", "lg", "xl") for i in range(1, 7)],
            "pvcnn.s3dis-area5.mit-han-lab",
            "pvcnn2.s3dis-area5",
        ],
    ),
    *_cases(
        "segmentation",
        "shapenetpart",
        [
            "dgcnn.shapenetpart.an-tao",
            "point-m2ae-base.shapenetpart.renrui-zhang",
            "point-mae-base.shapenetpart.yatian-pang",
            "pointnet.shapenetpart",
            "pointnext-sm.shapenetpart.openpoints",
            "pointnext-sm-c64.shapenetpart.openpoints",
            "pointnext-sm-c160.shapenetpart.openpoints",
        ],
    ),
    *_cases(
        "segmentation",
        "semantickitti",
        [
            "randlanet.semantickitti.tsung-han-wu",
            "sphereformer.semantickitti",
            "spvcnn-30gmacs.semantickitti.mit-han-lab",
            "spvcnn-47gmacs.semantickitti.mit-han-lab",
            "spvcnn-119gmacs.semantickitti.mit-han-lab",
        ],
    ),
    *_cases("segmentation", "nuscenes_lidarseg", ["sphereformer.nuscenes"]),
]

EXEMPT: Dict[Tuple[str, str], str] = {
    ("classification", "octformer-base.modelnet40.octree-nn"): "octree contract; resamples mesh faces",
    ("segmentation", "octformer-base.scannet20.octree-nn"): "octree contract",
    ("segmentation", "octformer-base.scannet200.octree-nn"): "octree contract",
    ("segmentation", "dgcnn.scannet20.an-tao"): "block-tiled input from `tile_scannet_scene`",
    ("detection", "lion-mamba.nuscenes.zhe-liu"): "voxel-stack contract (`HardVoxelize` keeps `pos`)",
    ("detection", "pointpillars.kitti.openpcdet"): "voxel-stack contract (`HardVoxelize` keeps `pos`)",
    ("detection", "pointpillars-multihead.nuscenes.openpcdet"): "voxel-stack contract (`HardVoxelize` keeps `pos`)",
    ("detection", "second.kitti.openpcdet"): "voxel-stack contract (`HardVoxelize` keeps `pos`)",
    ("detection", "second-multihead.nuscenes.openpcdet"): "voxel-stack contract (`HardVoxelize` keeps `pos`)",
    ("detection", "voxelnext.nuscenes.openpcdet"): "voxel-stack contract (`HardVoxelize` keeps `pos`)",
    ("detection", "voxel-mamba.waymo"): "voxel-stack contract; no Waymo dataset in the library",
}

MAP_KEYS = (DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT, DataKeys.INVERSE)


def test_every_registered_transform_is_listed() -> None:
    registered = {
        (task, name)
        for task, entries in _REGISTERED_MODELS.items()
        for name, entry in entries.items()
        if entry["transform"] is not None
    }
    listed = {(case.values[0], case.values[1]) for case in CASES} | set(EXEMPT)
    assert listed == registered


@pytest.mark.parametrize("task,name,family", CASES)
def test_registered_transform_follows_sampling_contract(task: str, name: str, family: str) -> None:
    torch.manual_seed(0)
    sample = BUILDERS[family](torch.Generator().manual_seed(0))
    n_origin = sample[DataKeys.POS].shape[0]

    out = _REGISTERED_MODELS[task][name]["transform"](dict(sample))
    n = out[DataKeys.POS].shape[0]
    for key in (DataKeys.X, DataKeys.SEGMENT):
        if key in out:
            assert out[key].shape[0] == n, f"{key!r} has {out[key].shape[0]} rows, `pos` has {n}"

    if n != n_origin:
        assert out[DataKeys.ORIGIN_POS].shape[0] == n_origin
        for key, value in out.items():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == n_origin:
                assert key in MAP_KEYS, f"{key!r} is left at source resolution ({n_origin} rows, `pos` has {n})"
        has_inverse, has_index = DataKeys.INVERSE in out, DataKeys.INDEX in out
        assert has_inverse != has_index, "exactly one of `inverse` / `index` must be present"
        if has_inverse:
            inverse = out[DataKeys.INVERSE]
            assert inverse.dtype == torch.long and inverse.shape == (n_origin,)
            assert int(inverse.min()) >= 0 and int(inverse.max()) < n
        else:
            index = out[DataKeys.INDEX]
            assert index.dtype == torch.long and index.shape == (n,)
            assert int(index.min()) >= 0 and int(index.max()) < n_origin
        if DataKeys.SEGMENT in out:
            assert out[DataKeys.ORIGIN_SEGMENT].shape[0] == n_origin

    batch = collate([out, out])
    assert batch[DataKeys.BATCH].shape[0] == 2 * n
