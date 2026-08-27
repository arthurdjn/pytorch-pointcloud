"""Every registered transform leaves a fixture sample in the one sampling-key contract.

`pos`, `x`, `segment` share the predictor resolution; a pipeline that changes the number of points keeps the
source cloud under `origin_pos` (and `origin_segment` when it carries labels) and records exactly one row
map: `inverse` (source row to predictor row, voxelizers) or `index` (predictor row to source row, selection
samplers). No other per-point tensor may be left at source resolution, and the output must collate.
"""

from typing import Callable, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import Dataset

from torch_pointcloud.models._registry import _REGISTERED_MODELS, Task
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)

CASES: List[Tuple[Task, str, str]] = [
    # Pretraining backbones on ScanNet
    ("base", "concerto-tiny.pretrain.pointcept", "scannet20"),
    ("base", "concerto-small.pretrain.pointcept", "scannet20"),
    ("base", "concerto-base.pretrain.pointcept", "scannet20"),
    ("base", "concerto-large.pretrain.pointcept", "scannet20"),
    ("base", "sonata-base.pretrain.fair", "scannet20"),
    ("base", "spformer-unet.scannet", "scannet20"),
    ("base", "utonia.pretrain.pointcept", "scannet20"),
    # ModelNet40 based models
    ("classification", "dgcnn.modelnet40-1024.an-tao", "modelnet_resampled"),
    ("classification", "dgcnn.modelnet40-2048.an-tao", "modelnet_resampled"),
    ("classification", "point-bert-base.modelnet40.xumin-yu", "modelnet_resampled"),
    ("classification", "point-bert-base.modelnet40-4k.xumin-yu", "modelnet_resampled"),
    ("classification", "point-bert-base.modelnet40-8k.xumin-yu", "modelnet_resampled"),
    ("classification", "point-m2ae-base.modelnet40.renrui-zhang", "modelnet_resampled"),
    ("classification", "point-mae-base.modelnet40.yatian-pang", "modelnet_resampled"),
    ("classification", "point-mae-base.modelnet40-8k.yatian-pang", "modelnet_resampled"),
    ("classification", "point-mamba-base.modelnet40.dingkang-liang", "modelnet_resampled"),
    ("classification", "point-transformer.modelnet40", "modelnet_resampled"),
    ("classification", "pointconv-density-base.modelnet40.wenxuan-wu", "modelnet_resampled"),
    *[("classification", f"pointgpt-{s}.modelnet40.guangyan-chen", "modelnet_resampled") for s in ("s", "b", "l")],
    *[("classification", f"pointgpt-{s}.modelnet40-8k.guangyan-chen", "modelnet_resampled") for s in ("s", "b", "l")],
    ("classification", "pointmlp-base.modelnet40.xu-ma", "modelnet_resampled"),
    ("classification", "pointmlp-elite.modelnet40.xu-ma", "modelnet_resampled"),
    ("classification", "pointnet.modelnet40", "modelnet_resampled"),
    ("classification", "pointnet2-msg.modelnet40.xu-yan", "modelnet_resampled"),
    ("classification", "pointnet2-ssg.modelnet40.xu-yan", "modelnet_resampled"),
    ("classification", "pointnet2.modelnet40.openpoints", "modelnet_resampled"),
    ("classification", "pointnext-sm-c64.modelnet40.openpoints", "modelnet_resampled"),
    # ScanObjectNN based models
    ("classification", "point-bert-base.scanobjectnn-hardest.xumin-yu", "scanobjectnn"),
    ("classification", "point-bert-base.scanobjectnn-objbg.xumin-yu", "scanobjectnn"),
    ("classification", "point-bert-base.scanobjectnn-objonly.xumin-yu", "scanobjectnn"),
    ("classification", "point-m2ae-base.scanobjectnn-hardest.renrui-zhang", "scanobjectnn"),
    ("classification", "point-m2ae-base.scanobjectnn-objbg.renrui-zhang", "scanobjectnn"),
    ("classification", "point-mamba-base.scanobjectnn.dingkang-liang", "scanobjectnn"),
    ("classification", "point-mamba-base.scanobjectnn-nobg.dingkang-liang", "scanobjectnn"),
    ("classification", "point-mamba-base.scanobjectnn-augmentedrot-scale75.dingkang-liang", "scanobjectnn"),
    *[
        ("classification", f"pointgpt-{s}.scanobjectnn-{v}.guangyan-chen", "scanobjectnn")
        for s in ("s", "b", "l")
        for v in ("hardest", "objbg", "objonly")
    ],
    ("classification", "pointmlp-base.scanobjectnn.xu-ma", "scanobjectnn"),
    ("classification", "pointmlp-elite.scanobjectnn.xu-ma", "scanobjectnn"),
    ("classification", "pointnet2.scanobjectnn.openpoints", "scanobjectnn"),
    ("classification", "pointnext-sm.scanobjectnn.openpoints", "scanobjectnn"),
    # Detection
    ("detection", "3detr.scannet.fair", "scannet20"),
    ("detection", "3detr-m.scannet.fair", "scannet20"),
    ("detection", "votenet.scannet.fair", "scannet20"),
    ("detection", "3detr.sunrgbd.fair", "sunrgbd"),
    ("detection", "votenet.sunrgbd.fair", "sunrgbd"),
    ("detection", "pointrcnn.kitti.openpcdet", "kitti"),
    # ScanNet based segmentation models
    ("segmentation", "concerto-large-lp.scannet20.pointcept", "scannet20"),
    ("segmentation", "oneformer3d-base.scannet20.danila-rukhovich", "scannet20"),
    ("segmentation", "oneformer3d-base.scannet200.danila-rukhovich", "scannet20"),
    ("segmentation", "point-transformer.scannet20", "scannet20"),
    ("segmentation", "ptv2-base.scannet20", "scannet20"),
    ("segmentation", "ptv2-base.scannet200", "scannet20"),
    ("segmentation", "ptv3-base.scannet20.pointcept", "scannet20"),
    ("segmentation", "ptv3-base.scannet200.pointcept", "scannet20"),
    ("segmentation", "sonata-lp.scannet20.fair", "scannet20"),
    ("segmentation", "spformer-unet.scannet20", "scannet20"),
    ("segmentation", "spunet-v1m1.scannet20.pointcept", "scannet20"),
    ("segmentation", "utonia-lp.scannet20.pointcept", "scannet20"),
    # S3DIS rooms
    ("segmentation", "oneformer3d-base.s3dis-area5.danila-rukhovich", "s3dis"),
    ("segmentation", "point-transformer.s3dis-area5", "s3dis"),
    *[("segmentation", f"pointnet2.s3dis-area{i}.openpoints", "s3dis") for i in range(1, 7)],
    ("segmentation", "ptv3-base.s3dis-area5.pointcept", "s3dis"),
    # S3DIS blocks
    *[("segmentation", f"dgcnn.s3dis-area{i}.an-tao", "s3dis_hdf5") for i in range(1, 7)],
    ("segmentation", "kpfcnn-base.s3dis.hugues-thomas", "s3dis_hdf5"),
    ("segmentation", "kpfcnn-base-sm.s3dis.hugues-thomas", "s3dis_hdf5"),
    ("segmentation", "kpfcnn-base-deform.s3dis.hugues-thomas", "s3dis_hdf5"),
    ("segmentation", "kpfcnn-base-sm-deform.s3dis.hugues-thomas", "s3dis_hdf5"),
    ("segmentation", "pointnet.s3dis-area5", "s3dis_hdf5"),
    ("segmentation", "pointnet2.s3dis-area5.xu-yan", "s3dis_hdf5"),
    *[
        ("segmentation", f"pointnext-{s}.s3dis-area{i}.openpoints", "s3dis_hdf5")
        for s in ("sm", "base", "lg", "xl")
        for i in range(1, 7)
    ],
    ("segmentation", "pvcnn.s3dis-area5.mit-han-lab", "s3dis_hdf5"),
    ("segmentation", "pvcnn2.s3dis-area5", "s3dis_hdf5"),
    # ShapeNetPart
    ("segmentation", "dgcnn.shapenetpart.an-tao", "shapenetpart"),
    ("segmentation", "point-m2ae-base.shapenetpart.renrui-zhang", "shapenetpart"),
    ("segmentation", "point-mae-base.shapenetpart.yatian-pang", "shapenetpart"),
    ("segmentation", "pointnet.shapenetpart", "shapenetpart"),
    ("segmentation", "pointnext-sm.shapenetpart.openpoints", "shapenetpart"),
    ("segmentation", "pointnext-sm-c64.shapenetpart.openpoints", "shapenetpart"),
    ("segmentation", "pointnext-sm-c160.shapenetpart.openpoints", "shapenetpart"),
    # SemanticKITTI
    ("segmentation", "randlanet.semantickitti.tsung-han-wu", "semantickitti"),
    ("segmentation", "sphereformer.semantickitti", "semantickitti"),
    ("segmentation", "spvcnn-30gmacs.semantickitti.mit-han-lab", "semantickitti"),
    ("segmentation", "spvcnn-47gmacs.semantickitti.mit-han-lab", "semantickitti"),
    ("segmentation", "spvcnn-119gmacs.semantickitti.mit-han-lab", "semantickitti"),
]

EXEMPT: List[Tuple[Task, str]] = [
    # Octree pipelines resample mesh faces, so no row map back to the input exists.
    ("classification", "octformer-base.modelnet40.octree-nn"),
    ("segmentation", "octformer-base.scannet20.octree-nn"),
    ("segmentation", "octformer-base.scannet200.octree-nn"),
    # Block-tiled input from `tile_scannet_scene`.
    ("segmentation", "dgcnn.scannet20.an-tao"),
    # `HardVoxelize` keeps `pos` and writes a voxel stack collated through `cat_keys`.
    ("detection", "lion-mamba.nuscenes.zhe-liu"),
    ("detection", "pointpillars.kitti.openpcdet"),
    ("detection", "pointpillars-multihead.nuscenes.openpcdet"),
    ("detection", "second.kitti.openpcdet"),
    ("detection", "second-multihead.nuscenes.openpcdet"),
    ("detection", "voxelnext.nuscenes.openpcdet"),
    ("detection", "voxel-mamba.waymo"),
    # The nuScenes loader ships no lidarseg labels.
    ("segmentation", "sphereformer.nuscenes"),
]


def test_every_registered_transform_is_listed() -> None:
    registered = {
        (task, name)
        for task, entries in _REGISTERED_MODELS.items()
        for name, entry in entries.items()
        if entry["transform"] is not None
    }
    assert {(task, name) for task, name, _ in CASES} | set(EXEMPT) == registered


@pytest.mark.parametrize("task,name,dataset_name", CASES, ids=[name for _, name, _ in CASES])
def test_registered_transform_follows_sampling_contract(
    task: Task, name: str, dataset_name: str, fixture_datasets: Dict[str, Callable[..., Dataset]]
) -> None:
    transform = _REGISTERED_MODELS[task][name]["transform"]
    assert transform is not None
    sample = fixture_datasets[dataset_name]()[0]
    n_origin = sample[DataKeys.POS].shape[0]

    out = transform(dict(sample))
    n = out[DataKeys.POS].shape[0]
    for key in (DataKeys.X, DataKeys.SEGMENT):
        if key in out:
            assert out[key].shape[0] == n, f"{key!r} has {out[key].shape[0]} rows, `pos` has {n}"

    if n != n_origin:
        assert out[DataKeys.ORIGIN_POS].shape[0] == n_origin
        for key, value in out.items():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == n_origin:
                assert key in (DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT, DataKeys.INVERSE), (
                    f"{key!r} is left at source resolution ({n_origin} rows, `pos` has {n})"
                )
        assert (DataKeys.INVERSE in out) != (DataKeys.INDEX in out), "exactly one of `inverse` / `index`"
        if DataKeys.INVERSE in out:
            inverse = out[DataKeys.INVERSE]
            assert inverse.dtype == torch.long and inverse.shape == (n_origin,)
            assert int(inverse.min()) >= 0 and int(inverse.max()) < n
        else:
            index = out[DataKeys.INDEX]
            assert index.dtype == torch.long and index.shape == (n,)
            assert int(index.min()) >= 0 and int(index.max()) < n_origin
        if DataKeys.SEGMENT in out:
            assert out[DataKeys.ORIGIN_SEGMENT].shape[0] == n_origin

    assert collate([out, out])[DataKeys.BATCH].shape[0] == 2 * n
