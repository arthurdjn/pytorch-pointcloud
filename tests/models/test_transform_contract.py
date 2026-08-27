"""Every registered transform leaves a fixture sample in the one sampling-key contract.

One case per distinct pipeline; the remaining registered names reuse one of those pipelines and are checked
against the listed cases by structure.

`pos`, `x`, `segment` share the predictor resolution; a pipeline that changes the number of points keeps the
source cloud under `origin_pos` (and `origin_segment` when it carries labels) and records exactly one row
map: `inverse` (source row to predictor row, voxelizers) or `index` (predictor row to source row, selection
samplers). No other per-point tensor may be left at source resolution, and the output must collate.
"""

from typing import Callable, List, Tuple

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
    # One row per distinct pipeline; `test_every_registered_transform_is_covered` maps the other names onto them.
    # Pretraining backbones on ScanNet
    ("base", "concerto-tiny.pretrain.pointcept", "scannet20"),
    ("base", "spformer-unet.scannet", "scannet20"),
    ("base", "utonia.pretrain.pointcept", "scannet20"),
    # ModelNet40 classification
    ("classification", "dgcnn.modelnet40-1024.an-tao", "modelnet"),
    ("classification", "dgcnn.modelnet40-2048.an-tao", "modelnet"),
    ("classification", "point-bert-base.modelnet40.xumin-yu", "modelnet"),
    ("classification", "point-bert-base.modelnet40-4k.xumin-yu", "modelnet"),
    ("classification", "point-bert-base.modelnet40-8k.xumin-yu", "modelnet"),
    ("classification", "point-mae-base.modelnet40-8k.yatian-pang", "modelnet"),
    ("classification", "point-transformer.modelnet40", "modelnet"),
    ("classification", "pointconv-density-base.modelnet40.wenxuan-wu", "modelnet"),
    ("classification", "pointmlp-base.modelnet40.xu-ma", "modelnet"),
    ("classification", "pointnet2-ssg.modelnet40.xu-yan", "modelnet"),
    ("classification", "pointnet2.modelnet40.openpoints", "modelnet"),
    # ScanObjectNN classification
    ("classification", "point-bert-base.scanobjectnn-objonly.xumin-yu", "scanobjectnn"),
    ("classification", "point-m2ae-base.scanobjectnn-hardest.renrui-zhang", "scanobjectnn"),
    ("classification", "point-mamba-base.scanobjectnn.dingkang-liang", "scanobjectnn"),
    ("classification", "pointnet2.scanobjectnn.openpoints", "scanobjectnn"),
    ("classification", "pointnext-sm.scanobjectnn.openpoints", "scanobjectnn"),
    # Detection
    ("detection", "3detr-m.scannet.fair", "scannet20"),
    ("detection", "3detr.sunrgbd.fair", "sunrgbd"),
    ("detection", "votenet.scannet.fair", "scannet20"),
    ("detection", "votenet.sunrgbd.fair", "sunrgbd"),
    ("detection", "pointrcnn.kitti.openpcdet", "kitti"),
    # ScanNet segmentation
    ("segmentation", "concerto-large-lp.scannet20.pointcept", "scannet20"),
    ("segmentation", "oneformer3d-base.scannet20.danila-rukhovich", "scannet20"),
    ("segmentation", "point-transformer.scannet20", "scannet20"),
    ("segmentation", "ptv2-base.scannet200", "scannet20"),
    ("segmentation", "ptv3-base.scannet20.pointcept", "scannet20"),
    ("segmentation", "ptv3-base.scannet200.pointcept", "scannet20"),
    ("segmentation", "spformer-unet.scannet20", "scannet20"),
    ("segmentation", "utonia-lp.scannet20.pointcept", "scannet20"),
    # S3DIS rooms
    ("segmentation", "oneformer3d-base.s3dis-area5.danila-rukhovich", "s3dis"),
    ("segmentation", "point-transformer.s3dis-area5", "s3dis"),
    ("segmentation", "pointnet2.s3dis-area1.openpoints", "s3dis"),
    ("segmentation", "ptv3-base.s3dis-area5.pointcept", "s3dis"),
    # S3DIS blocks
    ("segmentation", "dgcnn.s3dis-area1.an-tao", "s3dis_hdf5"),
    ("segmentation", "kpfcnn-base-sm.s3dis.hugues-thomas", "s3dis_hdf5"),
    ("segmentation", "pointnet2.s3dis-area5.xu-yan", "s3dis_hdf5"),
    ("segmentation", "pointnext-sm.s3dis-area1.openpoints", "s3dis_hdf5"),
    # ShapeNetPart
    ("segmentation", "dgcnn.shapenetpart.an-tao", "shapenetpart"),
    ("segmentation", "pointnet.shapenetpart", "shapenetpart"),
    ("segmentation", "pointnext-sm.shapenetpart.openpoints", "shapenetpart"),
    # SemanticKITTI
    ("segmentation", "randlanet.semantickitti.tsung-han-wu", "semantickitti"),
    ("segmentation", "sphereformer.semantickitti", "semantickitti"),
    ("segmentation", "spvcnn-30gmacs.semantickitti.mit-han-lab", "semantickitti"),
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


def test_every_registered_transform_is_covered() -> None:
    covered = {repr(_REGISTERED_MODELS[task][name]["transform"]) for task, name, _ in CASES}
    for task, entries in _REGISTERED_MODELS.items():
        for name, entry in entries.items():
            if entry["transform"] is not None and (task, name) not in EXEMPT:
                assert repr(entry["transform"]) in covered, f"{name!r} has a pipeline no listed case covers"


@pytest.mark.parametrize("task,name,dataset_name", CASES, ids=[name for _, name, _ in CASES])
def test_registered_transform_follows_sampling_contract(
    task: Task, name: str, dataset_name: str, dataset_factory: Callable[..., Dataset]
) -> None:
    transform = _REGISTERED_MODELS[task][name]["transform"]
    assert transform is not None
    sample = dataset_factory(dataset_name)[0]
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
