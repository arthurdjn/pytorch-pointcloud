"""Run every example script end to end on the dummy datasets shipped under `tests/data/datasets`."""

import ast
import importlib.util
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, List, Tuple

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _DWCONV_AVAILABLE,
    _FLASH_ATTN_AVAILABLE,
    _LASPY_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _OCNN_AVAILABLE,
    _PYG_LIB_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCHSPARSE_AVAILABLE,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
DATASETS_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets"
TIMEOUT = 900

_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
_REQUIRES_SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_TORCHSPARSE = pytest.mark.skipif(not _TORCHSPARSE_AVAILABLE, reason="torchsparse is not installed")
_REQUIRES_OCNN = pytest.mark.skipif(not _OCNN_AVAILABLE, reason="ocnn is not installed")
_REQUIRES_DWCONV = pytest.mark.skipif(not _DWCONV_AVAILABLE, reason="dwconv is not installed")
_REQUIRES_MAMBA = pytest.mark.skipif(not _MAMBA_SSM_AVAILABLE, reason="mamba-ssm is not installed")
_REQUIRES_PYG_LIB = pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
_REQUIRES_LASPY = pytest.mark.skipif(not _LASPY_AVAILABLE, reason="laspy is not installed")
_REQUIRES_FLASH_ATTN = pytest.mark.skipif(not _FLASH_ATTN_AVAILABLE, reason="flash-attn is not installed")

_CLUSTER = (_REQUIRES_PYG_LIB,)
_GPU_SPCONV = (_REQUIRES_SPCONV, _REQUIRES_CUDA)
_GPU_TORCHSPARSE = (_REQUIRES_TORCHSPARSE, _REQUIRES_CUDA)
_GPU_OCTREE = (_REQUIRES_OCNN, _REQUIRES_DWCONV, _REQUIRES_CUDA)
_GPU_MAMBA = (_REQUIRES_MAMBA, _REQUIRES_CUDA, _REQUIRES_PYG_LIB)
_GPU_FLASH_ATTN = (_REQUIRES_FLASH_ATTN, _REQUIRES_CUDA)
_LOADS_PRETRAINED_ENCODER = pytest.mark.pretrained

# Benchmarks load registry weights, so they also carry the `pretrained` marker. Checkpoints without a dummy
# dataset here (SUN RGB-D, ScanNet200, the KITTI split files) and the unreleased SphereFormer weights have no row.
BENCHMARKS = [
    pytest.param("spunet/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_SPCONV, id="spunet/scannet"),
    pytest.param(
        "spformer_unet/segmentation_scannet_benchmark.py",
        ("--limit", "1"),
        marks=_GPU_SPCONV,
        id="spformer_unet/scannet",
    ),
    pytest.param("kpconv/segmentation_s3dis_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="kpconv/s3dis"),
    pytest.param(
        "kpconv/segmentation_semantickitti_benchmark.py",
        ("--limit", "1"),
        marks=_CLUSTER,
        id="kpconv/semantickitti",
    ),
    pytest.param("kpconv/segmentation_toronto3d_benchmark.py", (), marks=_CLUSTER, id="kpconv/toronto3d"),
    pytest.param("kpconv/segmentation_parislille3d_benchmark.py", (), marks=_CLUSTER, id="kpconv/parislille3d"),
    pytest.param("pointnext/segmentation_s3dis_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="pointnext/s3dis"),
    pytest.param(
        "pointnext/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "4"),
        marks=_CLUSTER,
        id="pointnext/shapenetpart",
    ),
    pytest.param(
        "pointnext/classification_scanobjectnn_benchmark.py",
        ("--model", "pointnext-sm.scanobjectnn-hardest.openpoints", "--limit", "8"),
        marks=_CLUSTER,
        id="pointnext/scanobjectnn",
    ),
    pytest.param("pointnet/segmentation_s3dis_benchmark.py", ("--limit", "1"), id="pointnet/s3dis"),
    pytest.param("pointnet/classification_modelnet40_benchmark.py", ("--limit", "4"), id="pointnet/modelnet40"),
    pytest.param(
        "pointnet/classification_scanobjectnn_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="pointnet/scanobjectnn",
    ),
    pytest.param(
        "pointnet2/segmentation_s3dis_benchmark.py",
        ("--model", "pointnet2.s3dis-area5.xu-yan", "--limit", "1"),
        marks=_CLUSTER,
        id="pointnet2/s3dis-xu-yan",
    ),
    pytest.param(
        "pointnet2/segmentation_s3dis_benchmark.py",
        ("--model", "pointnet2.s3dis-area5.openpoints", "--limit", "1"),
        marks=_CLUSTER,
        id="pointnet2/s3dis-openpoints",
    ),
    pytest.param(
        "pointnet2/classification_modelnet40_benchmark.py",
        ("--model", "pointnet2-msg.modelnet40.xu-yan", "--limit", "8"),
        marks=_CLUSTER,
        id="pointnet2/modelnet40",
    ),
    pytest.param(
        "pointnet2/classification_scanobjectnn_benchmark.py",
        ("--model", "pointnet2.scanobjectnn-hardest.openpoints", "--limit", "8"),
        marks=_CLUSTER,
        id="pointnet2/scanobjectnn",
    ),
    pytest.param("pvcnn/segmentation_s3dis_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="pvcnn/s3dis"),
    pytest.param(
        "pvcnn/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "4", "--num-votes", "2"),
        marks=_CLUSTER,
        id="pvcnn/shapenetpart",
    ),
    pytest.param(
        "pointnet2/part_segmentation_shapenetpart_benchmark.py",
        ("--model", "pointnet2-msg.shapenetpart.mit-han-lab", "--limit", "4", "--num-votes", "2"),
        marks=_CLUSTER,
        id="pointnet2/shapenetpart",
    ),
    pytest.param("sonata/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_SPCONV, id="sonata/scannet"),
    pytest.param(
        "concerto/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_SPCONV, id="concerto/scannet"
    ),
    pytest.param("utonia/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_SPCONV, id="utonia/scannet"),
    pytest.param(
        "point_transformer_v3/segmentation_scannet_benchmark.py",
        ("--model", "ptv3-base.scannet20.pointcept", "--limit", "1"),
        marks=_GPU_SPCONV,
        id="ptv3/scannet",
    ),
    pytest.param(
        "point_transformer_v3/segmentation_dales_benchmark.py",
        ("--model", "ptv3.dales.jayakumar-pujar", "--limit", "1"),
        marks=_GPU_SPCONV,
        id="ptv3/dales",
    ),
    pytest.param(
        "point_transformer_v3/segmentation_nuscenes_benchmark.py",
        ("--model", "ptv3-base.nuscenes.pointcept", "--split", "mini", "--limit", "1"),
        marks=_GPU_SPCONV,
        id="ptv3/nuscenes",
    ),
    pytest.param(
        "point_transformer_v3/segmentation_nuscenes_benchmark.py",
        ("--model", "ptv3-base.nuscenes.pointcept", "--split", "mini", "--limit", "1", "--simple"),
        marks=_GPU_SPCONV,
        id="ptv3/nuscenes-simple",
    ),
    pytest.param("volt/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_FLASH_ATTN, id="volt/scannet"),
    pytest.param(
        "volt/segmentation_scannet_benchmark.py",
        ("--limit", "1", "--simple"),
        marks=_GPU_FLASH_ATTN,
        id="volt/scannet-simple",
    ),
    pytest.param(
        "volt/segmentation_nuscenes_benchmark.py",
        ("--split", "mini", "--limit", "1"),
        marks=_GPU_FLASH_ATTN,
        id="volt/nuscenes",
    ),
    pytest.param(
        "volt/segmentation_nuscenes_benchmark.py",
        ("--split", "mini", "--limit", "1", "--simple"),
        marks=_GPU_FLASH_ATTN,
        id="volt/nuscenes-simple",
    ),
    pytest.param(
        "volt/segmentation_semantickitti_benchmark.py",
        ("--limit", "1"),
        marks=_GPU_FLASH_ATTN,
        id="volt/semantickitti",
    ),
    pytest.param(
        "volt/segmentation_semantickitti_benchmark.py",
        ("--limit", "1", "--simple"),
        marks=_GPU_FLASH_ATTN,
        id="volt/semantickitti-simple",
    ),
    pytest.param(
        "point_transformer_v3/segmentation_gridnethd_benchmark.py",
        ("--model", "ptv3.gridnet-hd.heig-vd-geo", "--limit", "1"),
        marks=(*_GPU_SPCONV, _REQUIRES_LASPY),
        id="ptv3/gridnethd",
    ),
    pytest.param(
        "litept/segmentation_scannet_benchmark.py",
        ("--model", "litept-small.scannet20.prs-eth", "--limit", "1"),
        marks=_GPU_SPCONV,
        id="litept/scannet",
    ),
    pytest.param(
        "litept/segmentation_nuscenes_benchmark.py",
        ("--model", "litept-small.nuscenes.prs-eth", "--split", "mini", "--limit", "1"),
        marks=_GPU_SPCONV,
        id="litept/nuscenes",
    ),
    pytest.param(
        "litept/segmentation_nuscenes_benchmark.py",
        ("--model", "litept-small.nuscenes.prs-eth", "--split", "mini", "--limit", "1", "--simple"),
        marks=_GPU_SPCONV,
        id="litept/nuscenes-simple",
    ),
    pytest.param(
        "point_transformer/segmentation_s3dis_benchmark.py",
        ("--limit", "1"),
        marks=_CLUSTER,
        id="point_transformer/s3dis",
    ),
    pytest.param(
        "randlanet/segmentation_fractal_benchmark.py",
        ("--limit", "2"),
        marks=(_REQUIRES_PYG_LIB, _REQUIRES_LASPY),
        id="randlanet/fractal",
    ),
    pytest.param(
        "randlanet/segmentation_semantickitti_benchmark.py",
        ("--limit", "1"),
        marks=_CLUSTER,
        id="randlanet/semantickitti",
    ),
    pytest.param("randlanet/segmentation_s3dis_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="randlanet/s3dis"),
    pytest.param("randlanet/segmentation_toronto3d_benchmark.py", (), marks=_CLUSTER, id="randlanet/toronto3d"),
    pytest.param("randlanet/segmentation_parislille3d_benchmark.py", (), marks=_CLUSTER, id="randlanet/parislille3d"),
    pytest.param(
        "randlanet/segmentation_semantic3d_benchmark.py",
        ("--scenes", "bildstein_station1_xyz_intensity_rgb"),
        marks=_CLUSTER,
        id="randlanet/semantic3d",
    ),
    pytest.param(
        "minkunet/segmentation_semantickitti_benchmark.py",
        ("--limit", "1"),
        marks=_GPU_SPCONV,
        id="minkunet/semantickitti",
    ),
    pytest.param(
        "spvcnn/segmentation_semantickitti_benchmark.py",
        ("--limit", "1"),
        marks=_GPU_TORCHSPARSE,
        id="spvcnn/semantickitti",
    ),
    pytest.param(
        "cylinder3d/segmentation_semantickitti_benchmark.py",
        ("--limit", "1"),
        marks=_GPU_SPCONV,
        id="cylinder3d/semantickitti",
    ),
    pytest.param(
        "cylinder3d/segmentation_nuscenes_benchmark.py",
        ("--split", "mini", "--limit", "1"),
        marks=_GPU_SPCONV,
        id="cylinder3d/nuscenes",
    ),
    pytest.param("dgcnn/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="dgcnn/scannet"),
    pytest.param(
        "dgcnn/segmentation_s3dis_benchmark.py",
        ("--model", "dgcnn.s3dis-area5.an-tao", "--limit", "4"),
        marks=_CLUSTER,
        id="dgcnn/s3dis",
    ),
    pytest.param(
        "dgcnn/part_segmentation_shapenetpart_benchmark.py", ("--limit", "4"), marks=_CLUSTER, id="dgcnn/shapenetpart"
    ),
    pytest.param("pct/classification_modelnet40_benchmark.py", ("--limit", "4"), marks=_CLUSTER, id="pct/modelnet40"),
    pytest.param(
        "curvenet/classification_modelnet40_benchmark.py", ("--limit", "4"), marks=_CLUSTER, id="curvenet/modelnet40"
    ),
    pytest.param(
        "curvenet/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "4"),
        marks=_CLUSTER,
        id="curvenet/shapenetpart",
    ),
    pytest.param(
        "octformer/segmentation_scannet_benchmark.py",
        ("--model", "octformer-base.scannet20.octree-nn", "--limit", "1"),
        marks=_GPU_OCTREE,
        id="octformer/scannet",
    ),
    pytest.param("octformer/classification_modelnet40_benchmark.py", (), marks=_GPU_OCTREE, id="octformer/modelnet40"),
    pytest.param(
        "point_bert/classification_modelnet40_benchmark.py",
        ("--model", "point-bert-base.modelnet40.xumin-yu", "--limit", "8"),
        marks=_CLUSTER,
        id="point_bert/modelnet40",
    ),
    pytest.param(
        "point_bert/classification_scanobjectnn_benchmark.py",
        ("--model", "point-bert-base.scanobjectnn-hardest.xumin-yu", "--limit", "8"),
        marks=_CLUSTER,
        id="point_bert/scanobjectnn",
    ),
    pytest.param(
        "point_mae/classification_scanobjectnn_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point_mae/scanobjectnn",
    ),
    pytest.param(
        "point_m2ae/classification_scanobjectnn_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point_m2ae/scanobjectnn",
    ),
    pytest.param(
        "point_mamba/classification_scanobjectnn_benchmark.py",
        ("--limit", "8"),
        marks=_GPU_MAMBA,
        id="point_mamba/scanobjectnn",
    ),
    pytest.param(
        "point_mae/classification_modelnet40_benchmark.py",
        ("--model", "point-mae-base.modelnet40.yatian-pang", "--limit", "8"),
        marks=_CLUSTER,
        id="point_mae/modelnet40",
    ),
    pytest.param(
        "point_mae/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point_mae/shapenetpart",
    ),
    pytest.param(
        "recon/classification_scanobjectnn_benchmark.py",
        ("--model", "recon.scanobjectnn-hardest.zekun-qi", "--limit", "8"),
        marks=_CLUSTER,
        id="recon/scanobjectnn",
    ),
    pytest.param(
        "recon/classification_modelnet40_benchmark.py",
        ("--model", "recon.modelnet40.zekun-qi", "--limit", "8"),
        marks=_CLUSTER,
        id="recon/modelnet40",
    ),
    pytest.param(
        "recon/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="recon/shapenetpart",
    ),
    pytest.param(
        "act/classification_scanobjectnn_benchmark.py",
        ("--model", "act.scanobjectnn-hardest.runpei-dong", "--limit", "8"),
        marks=_CLUSTER,
        id="act/scanobjectnn",
    ),
    pytest.param(
        "act/classification_modelnet40_benchmark.py",
        ("--model", "act.modelnet40.runpei-dong", "--limit", "8"),
        marks=_CLUSTER,
        id="act/modelnet40",
    ),
    pytest.param(
        "pcp_mae/classification_scanobjectnn_benchmark.py",
        ("--model", "pcp-mae.scanobjectnn-hardest.xiangdong-zhang", "--limit", "8"),
        marks=_CLUSTER,
        id="pcp_mae/scanobjectnn",
    ),
    pytest.param(
        "pcp_mae/classification_modelnet40_benchmark.py",
        ("--model", "pcp-mae.modelnet40.xiangdong-zhang", "--limit", "8"),
        marks=_CLUSTER,
        id="pcp_mae/modelnet40",
    ),
    pytest.param(
        "point2vec/classification_scanobjectnn_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point2vec/scanobjectnn",
    ),
    pytest.param(
        "point2vec/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point2vec/shapenetpart",
    ),
    pytest.param(
        "point_m2ae/classification_modelnet40_benchmark.py",
        ("--model", "point-m2ae-base.modelnet40.renrui-zhang", "--limit", "8"),
        marks=_CLUSTER,
        id="point_m2ae/modelnet40",
    ),
    pytest.param(
        "point_m2ae/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point_m2ae/shapenetpart",
    ),
    pytest.param(
        "pointgpt/classification_scanobjectnn_benchmark.py",
        ("--model", "pointgpt-s.scanobjectnn-objonly.guangyan-chen", "--limit", "8"),
        marks=_CLUSTER,
        id="pointgpt/scanobjectnn",
    ),
    pytest.param(
        "pointgpt/part_segmentation_shapenetpart_benchmark.py",
        ("--model", "pointgpt-s.shapenetpart.guangyan-chen", "--limit", "8"),
        marks=_CLUSTER,
        id="pointgpt/shapenetpart",
    ),
    pytest.param(
        "point_mamba/classification_modelnet40_benchmark.py",
        ("--model", "point-mamba-base.modelnet40.dingkang-liang", "--limit", "8"),
        marks=_GPU_MAMBA,
        id="point_mamba/modelnet40",
    ),
    pytest.param(
        "pointmlp/classification_scanobjectnn_benchmark.py",
        ("--model", "pointmlp-base.scanobjectnn-hardest.xu-ma", "--limit", "8"),
        marks=_CLUSTER,
        id="pointmlp/scanobjectnn",
    ),
    pytest.param(
        "pointconv/classification_modelnet40_benchmark.py", ("--limit", "8"), marks=_CLUSTER, id="pointconv/modelnet40"
    ),
    pytest.param(
        "votenet/detection_scannet_benchmark.py",
        ("--model", "votenet.scannet.fair", "--limit", "2"),
        marks=_CLUSTER,
        id="votenet/scannet",
    ),
    pytest.param(
        "fcaf3d/detection_benchmark.py",
        ("--dataset", "scannet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="fcaf3d/scannet",
    ),
    pytest.param(
        "fcaf3d/detection_benchmark.py",
        ("--dataset", "sunrgbd", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="fcaf3d/sunrgbd",
    ),
    pytest.param(
        "fcaf3d/detection_benchmark.py",
        ("--dataset", "s3dis", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="fcaf3d/s3dis",
    ),
    pytest.param(
        "tr3d/detection_benchmark.py",
        ("--dataset", "scannet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="tr3d/scannet",
    ),
    pytest.param(
        "tr3d/detection_benchmark.py",
        ("--dataset", "sunrgbd", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="tr3d/sunrgbd",
    ),
    pytest.param(
        "tr3d/detection_benchmark.py",
        ("--dataset", "s3dis", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="tr3d/s3dis",
    ),
    pytest.param(
        "votenet/detection_scannet_benchmark.py",
        ("--model", "votenet.scannet.mmdet3d", "--limit", "2"),
        marks=_CLUSTER,
        id="votenet/scannet-mmdet3d",
    ),
    pytest.param(
        "votenet/detection_sunrgbd_benchmark.py",
        ("--model", "votenet.sunrgbd.mmdet3d", "--limit", "2"),
        marks=_CLUSTER,
        id="votenet/sunrgbd-mmdet3d",
    ),
    pytest.param(
        "threedetr/detection_sunrgbd_benchmark.py",
        ("--model", "3detr-m.sunrgbd.fair", "--limit", "2"),
        marks=_CLUSTER,
        id="3detr/sunrgbd",
    ),
    pytest.param(
        "threedetr/detection_scannet_benchmark.py",
        ("--model", "3detr-m.scannet.fair", "--limit", "2"),
        marks=_CLUSTER,
        id="3detr/scannet",
    ),
    pytest.param(
        "groupfree3d/detection_scannet_benchmark.py",
        ("--model", "groupfree3d-l6-o256.scannet.ze-liu", "--limit", "2"),
        marks=_CLUSTER,
        id="groupfree3d/scannet",
    ),
    pytest.param(
        "groupfree3d/detection_sunrgbd_benchmark.py",
        ("--model", "groupfree3d-l6-o256.sunrgbd.ze-liu", "--limit", "2"),
        marks=_CLUSTER,
        id="groupfree3d/sunrgbd",
    ),
    pytest.param(
        "anchor_detectors/detection_kitti_openpcdet_benchmark.py",
        ("--model", "second.kitti.openpcdet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/second.kitti.openpcdet",
    ),
    pytest.param(
        "anchor_detectors/detection_nuscenes_openpcdet_benchmark.py",
        ("--model", "second-multihead.nuscenes.openpcdet", "--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/second-multihead.nuscenes.openpcdet",
    ),
    pytest.param(
        "anchor_detectors/detection_kitti_openpcdet_benchmark.py",
        ("--model", "pointpillars.kitti.openpcdet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/pointpillars.kitti.openpcdet",
    ),
    pytest.param(
        "anchor_detectors/detection_nuscenes_openpcdet_benchmark.py",
        ("--model", "pointpillars-multihead.nuscenes.openpcdet", "--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/pointpillars-multihead.nuscenes.openpcdet",
    ),
    pytest.param(
        "anchor_detectors/detection_nuscenes_mmdet3d_benchmark.py",
        ("--model", "pointpillars.nuscenes.mmdet3d", "--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/pointpillars.nuscenes.mmdet3d",
    ),
    pytest.param(
        "anchor_detectors/detection_nuscenes_mmdet3d_benchmark.py",
        ("--model", "pointpillars-fpn.nuscenes.mmdet3d", "--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/pointpillars-fpn.nuscenes.mmdet3d",
    ),
    pytest.param(
        "anchor_detectors/detection_kitti_mmdet3d_benchmark.py",
        ("--model", "pointpillars.kitti.mmdet3d", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/pointpillars.kitti.mmdet3d",
    ),
    pytest.param(
        "anchor_detectors/detection_kitti_mmdet3d_benchmark.py",
        ("--model", "pointpillars.kitti-car.mmdet3d", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/pointpillars.kitti-car.mmdet3d",
    ),
    pytest.param(
        "anchor_detectors/detection_kitti_mmdet3d_benchmark.py",
        ("--model", "second.kitti.mmdet3d", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/second.kitti.mmdet3d",
    ),
    pytest.param(
        "anchor_detectors/detection_kitti_mmdet3d_benchmark.py",
        ("--model", "second.kitti-car.mmdet3d", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/second.kitti-car.mmdet3d",
    ),
    pytest.param(
        "anchor_detectors/detection_kitti_mmdet3d_benchmark.py",
        ("--model", "second-dynamic-voxel.kitti.mmdet3d", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/second-dynamic-voxel.kitti.mmdet3d",
    ),
    pytest.param(
        "anchor_detectors/detection_kitti_openpcdet_benchmark.py",
        ("--model", "second-iou.kitti.openpcdet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="anchor_detectors/second-iou.kitti.openpcdet",
    ),
    pytest.param("pointrcnn/detection_kitti_benchmark.py", ("--limit", "2"), marks=_CLUSTER, id="pointrcnn/kitti"),
    pytest.param(
        "pointrcnn/detection_kitti_benchmark.py",
        ("--model", "pointrcnn-iou.kitti.openpcdet", "--limit", "2"),
        marks=_CLUSTER,
        id="pointrcnn/kitti-iou",
    ),
    pytest.param(
        "parta2/detection_kitti_benchmark.py",
        ("--model", "part-a2.kitti.mmdet3d", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="parta2/kitti",
    ),
    pytest.param(
        "parta2/detection_kitti_benchmark.py",
        ("--model", "part-a2.kitti-car.mmdet3d", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="parta2/kitti-car",
    ),
    pytest.param(
        "pv_rcnn/detection_kitti_benchmark.py",
        ("--model", "pv-rcnn.kitti.mmdet3d", "--limit", "2"),
        marks=(*_GPU_SPCONV, _REQUIRES_PYG_LIB),
        id="pv_rcnn/kitti",
    ),
    pytest.param(
        "voxel_rcnn/detection_kitti_benchmark.py",
        ("--model", "voxel-rcnn.kitti-car.openpcdet", "--limit", "2"),
        marks=(*_GPU_SPCONV, _REQUIRES_PYG_LIB),
        id="voxel_rcnn/kitti-car",
    ),
    pytest.param(
        "voxelnext/detection_nuscenes_benchmark.py",
        ("--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="voxelnext/nuscenes",
    ),
    pytest.param(
        "centerpoint/detection_nuscenes_benchmark.py",
        ("--model", "centerpoint-voxel01.nuscenes.mmdet3d", "--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="centerpoint/nuscenes",
    ),
    pytest.param(
        "centerpoint/detection_nuscenes_benchmark.py",
        ("--model", "centerpoint-pillar02.nuscenes.mmdet3d", "--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="centerpoint-pillars/nuscenes",
    ),
    pytest.param(
        "transfusion/detection_nuscenes_benchmark.py",
        ("--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="transfusion/nuscenes",
    ),
    pytest.param(
        "lion/detection_nuscenes_benchmark.py",
        ("--split", "mini", "--limit", "2"),
        marks=_GPU_MAMBA,
        id="lion/nuscenes",
    ),
]

# The reproduction scripts (one per reference recipe) validate on `--limit-val-batches` scenes.
_SMOKE_RECIPE = ("--limit-train-batches", "2", "--limit-val-batches", "2", "--epochs", "1", "--eval-every", "1")
# One classification and one segmentation dataset per training script; VoteNet has no SUN RGB-D dummy data.
TRAININGS = [
    pytest.param(
        "dgcnn/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="dgcnn/modelnet40",
    ),
    pytest.param(
        "dgcnn/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="dgcnn/shapenetpart",
    ),
    pytest.param(
        "pct/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pct/modelnet40",
    ),
    pytest.param(
        "curvenet/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="curvenet/modelnet40",
    ),
    pytest.param(
        "curvenet/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="curvenet/shapenetpart",
    ),
    pytest.param(
        "kpconv/segmentation_s3dis_train.py", (*_SMOKE_RECIPE, "--batch-size", "2"), marks=_CLUSTER, id="kpconv/s3dis"
    ),
    pytest.param(
        "kpconv/segmentation_s3dis_train.py",
        (*_SMOKE_RECIPE, "--model", "kpfcnn-base-sm-deform.s3dis-area5.hugues-thomas", "--batch-size", "2"),
        marks=_CLUSTER,
        id="kpconv/s3dis-deform",
    ),
    pytest.param(
        "point_mae/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point_mae/finetune",
    ),
    pytest.param(
        "recon/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="recon/finetune",
    ),
    pytest.param(
        "act/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="act/finetune",
    ),
    pytest.param(
        "pcp_mae/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="pcp_mae/finetune",
    ),
    pytest.param(
        "point2vec/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point2vec/finetune",
    ),
    pytest.param(
        "point_bert/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point_bert/finetune",
    ),
    pytest.param(
        "point_m2ae/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point_m2ae/finetune",
    ),
    pytest.param(
        "point_mamba/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_GPU_MAMBA, _LOADS_PRETRAINED_ENCODER),
        id="point_mamba/finetune",
    ),
    pytest.param(
        "pointgpt/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="pointgpt/scanobjectnn",
    ),
    pytest.param(
        "pointgpt/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="pointgpt/shapenetpart-train",
    ),
    pytest.param(
        "pointmlp/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointmlp/scanobjectnn",
    ),
    pytest.param(
        "pointmlp/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pointmlp/modelnet40",
    ),
    pytest.param(
        "pointnet2/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointnet2/modelnet40-ssg",
    ),
    pytest.param(
        "pointnet2/segmentation_s3dis_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointnet2/s3dis-xu-yan",
    ),
    pytest.param(
        "pointnet2/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--model", "pointnet2-msg.modelnet40.xu-yan", "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointnet2/modelnet40-msg",
    ),
    pytest.param(
        "pointnet/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        id="pointnet/modelnet40-jiawei-ren",
    ),
    pytest.param(
        "pointnet/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--model", "pointnet.modelnet40", "--batch-size", "2", "--val-batch-size", "2"),
        id="pointnet/modelnet40",
    ),
    pytest.param(
        "pointnet/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pointnet/scanobjectnn",
    ),
    pytest.param("pointnet/segmentation_s3dis_train.py", (*_SMOKE_RECIPE, "--batch-size", "2"), id="pointnet/s3dis"),
    pytest.param(
        "pointnext/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pointnext/scanobjectnn",
    ),
    pytest.param(
        "pointnet2/classification_scanobjectnn_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pointnet2/scanobjectnn-train",
    ),
    pytest.param(
        "pointnext/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pointnext/modelnet40",
    ),
    pytest.param(
        "pointnext/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--model", "pointnet2.modelnet40.openpoints", "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pointnet2/modelnet40-openpoints",
    ),
    pytest.param(
        "pointnext/segmentation_s3dis_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointnext/s3dis",
    ),
    pytest.param(
        "pointnext/segmentation_s3dis_train.py",
        (*_SMOKE_RECIPE, "--model", "pointnet2", "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointnet2/s3dis-openpoints",
    ),
    pytest.param(
        "pointnext/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointnext/shapenetpart",
    ),
    pytest.param(
        "spunet/segmentation_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="spunet/scannet",
    ),
    pytest.param(
        "sonata/segmentation_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_GPU_SPCONV, _LOADS_PRETRAINED_ENCODER),
        id="sonata/scannet-train",
    ),
    pytest.param(
        "concerto/segmentation_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_GPU_SPCONV, _LOADS_PRETRAINED_ENCODER),
        id="concerto/scannet-train",
    ),
    pytest.param(
        "utonia/segmentation_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_GPU_SPCONV, _LOADS_PRETRAINED_ENCODER),
        id="utonia/scannet-train",
    ),
    pytest.param(
        "point_transformer_v3/segmentation_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="point_transformer_v3/scannet",
    ),
    pytest.param(
        "point_transformer_v3/segmentation_s3dis_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="point_transformer_v3/s3dis",
    ),
    pytest.param(
        "litept/segmentation_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="litept/scannet-train",
    ),
    pytest.param(
        "pvcnn/segmentation_s3dis_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pvcnn/s3dis",
    ),
    pytest.param(
        "pvcnn/segmentation_s3dis_train.py",
        (*_SMOKE_RECIPE, "--model", "pvcnn2.s3dis-area5.mit-han-lab", "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pvcnn/s3dis-pvcnn2",
    ),
    pytest.param(
        "pvcnn/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2", "--val-batch-size", "2"),
        marks=_CLUSTER,
        id="pvcnn/shapenetpart",
    ),
    pytest.param(
        "pointnet2/part_segmentation_shapenetpart_train.py",
        (
            *_SMOKE_RECIPE,
            "--model",
            "pointnet2-msg.shapenetpart.mit-han-lab",
            "--batch-size",
            "2",
            "--val-batch-size",
            "2",
        ),
        marks=_CLUSTER,
        id="pointnet2/shapenetpart",
    ),
    pytest.param(
        "randlanet/segmentation_semantickitti_train.py",
        (
            *_SMOKE_RECIPE,
            "--train-sequences",
            "00",
            "--val-sequences",
            "08",
            "--batch-size",
            "2",
            "--val-batch-size",
            "2",
        ),
        marks=_CLUSTER,
        id="randlanet/semantickitti",
    ),
    pytest.param(
        "minkunet/segmentation_semantickitti_train.py",
        (
            *_SMOKE_RECIPE,
            "--train-sequences",
            "00",
            "--val-sequences",
            "08",
            "--batch-size",
            "2",
            "--warmup-iters",
            "1",
            "--max-voxels",
            "256",
        ),
        marks=_GPU_SPCONV,
        id="minkunet/semantickitti",
    ),
    pytest.param(
        "spvcnn/segmentation_semantickitti_train.py",
        (
            *_SMOKE_RECIPE,
            "--train-sequences",
            "00",
            "--val-sequences",
            "08",
            "--batch-size",
            "2",
            "--val-batch-size",
            "1",
            "--warmup-iters",
            "1",
        ),
        marks=_GPU_TORCHSPARSE,
        id="spvcnn/semantickitti",
    ),
    pytest.param(
        "cylinder3d/segmentation_semantickitti_train.py",
        (
            *_SMOKE_RECIPE,
            "--train-sequences",
            "00",
            "--val-sequences",
            "08",
            "--batch-size",
            "2",
            "--warmup-iters",
            "1",
        ),
        marks=_GPU_SPCONV,
        id="cylinder3d/semantickitti",
    ),
    pytest.param(
        "sphereformer/segmentation_semantickitti_train.py",
        (*_SMOKE_RECIPE, "--train-sequences", "00", "--val-sequences", "08", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="sphereformer/semantickitti",
    ),
    pytest.param(
        "octformer/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_OCTREE,
        id="octformer/modelnet40",
    ),
    pytest.param(
        "point_bert/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point_bert/modelnet40",
    ),
    pytest.param(
        "point_m2ae/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point_m2ae/modelnet40",
    ),
    pytest.param(
        "point_mae/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point_mae/modelnet40",
    ),
    pytest.param(
        "recon/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="recon/modelnet40",
    ),
    pytest.param(
        "act/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="act/modelnet40",
    ),
    pytest.param(
        "pcp_mae/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="pcp_mae/modelnet40",
    ),
    pytest.param(
        "point2vec/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point2vec/modelnet40",
    ),
    pytest.param(
        "point_mamba/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_GPU_MAMBA, _LOADS_PRETRAINED_ENCODER),
        id="point_mamba/modelnet40",
    ),
    pytest.param(
        "pointconv/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointconv/modelnet40",
    ),
    pytest.param(
        "pointgpt/classification_modelnet40_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="pointgpt/modelnet40",
    ),
    pytest.param(
        "threedetr/detection_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="threedetr/scannet",
    ),
    pytest.param(
        "threedetr/detection_sunrgbd_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="threedetr/sunrgbd",
    ),
    pytest.param(
        "groupfree3d/detection_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="groupfree3d/scannet-train",
    ),
    pytest.param(
        "groupfree3d/detection_sunrgbd_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="groupfree3d/sunrgbd-train",
    ),
    pytest.param(
        "votenet/detection_sunrgbd_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="votenet/sunrgbd-train",
    ),
    pytest.param(
        "votenet/detection_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="votenet/scannet-train",
    ),
    pytest.param(
        "octformer/segmentation_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_OCTREE,
        id="octformer/scannet-train",
    ),
    pytest.param(
        "octformer/segmentation_scannet200_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_OCTREE,
        id="octformer/scannet200-train",
    ),
    pytest.param(
        "point_transformer_v3/segmentation_scannet200_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="point_transformer_v3/scannet200",
    ),
    pytest.param(
        "dgcnn/segmentation_s3dis_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="dgcnn/s3dis-train",
    ),
    pytest.param(
        "dgcnn/segmentation_scannet_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="dgcnn/scannet-train",
    ),
    pytest.param(
        "point_mae/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point_mae/shapenetpart-train",
    ),
    pytest.param(
        "recon/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="recon/shapenetpart-train",
    ),
    pytest.param(
        "point2vec/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point2vec/shapenetpart-train",
    ),
    pytest.param(
        "point_m2ae/part_segmentation_shapenetpart_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_CLUSTER, _LOADS_PRETRAINED_ENCODER),
        id="point_m2ae/shapenetpart-train",
    ),
    pytest.param(
        "fcaf3d/detection_train.py",
        (*_SMOKE_RECIPE, "--dataset", "scannet", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="fcaf3d/scannet-train",
    ),
    pytest.param(
        "fcaf3d/detection_train.py",
        (*_SMOKE_RECIPE, "--dataset", "sunrgbd", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="fcaf3d/sunrgbd-train",
    ),
    pytest.param(
        "fcaf3d/detection_train.py",
        (*_SMOKE_RECIPE, "--dataset", "s3dis", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="fcaf3d/s3dis-train",
    ),
    pytest.param(
        "tr3d/detection_train.py",
        (*_SMOKE_RECIPE, "--dataset", "scannet", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="tr3d/scannet-train",
    ),
    pytest.param(
        "tr3d/detection_train.py",
        (*_SMOKE_RECIPE, "--dataset", "s3dis", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="tr3d/s3dis-train",
    ),
    pytest.param(
        "tr3d/detection_train.py",
        (*_SMOKE_RECIPE, "--dataset", "sunrgbd", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="tr3d/sunrgbd-train",
    ),
    pytest.param(
        "second/detection_kitti_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="second/kitti-train",
    ),
    pytest.param(
        "pointpillars/detection_kitti_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="pointpillars/kitti-train",
    ),
    pytest.param(
        "pointrcnn/detection_kitti_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=_CLUSTER,
        id="pointrcnn/kitti-train",
    ),
    pytest.param(
        "voxel_rcnn/detection_kitti_train.py",
        (*_SMOKE_RECIPE, "--batch-size", "2"),
        marks=(*_GPU_SPCONV, _REQUIRES_PYG_LIB),
        id="voxel_rcnn/kitti-train",
    ),
    pytest.param(
        "second/detection_nuscenes_train.py",
        (*_SMOKE_RECIPE, "--split", "mini", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="second/nuscenes-train",
    ),
    pytest.param(
        "pointpillars/detection_nuscenes_train.py",
        (*_SMOKE_RECIPE, "--split", "mini", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="pointpillars/nuscenes-train",
    ),
    pytest.param(
        "voxelnext/detection_nuscenes_train.py",
        (*_SMOKE_RECIPE, "--split", "mini", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="voxelnext/nuscenes-train",
    ),
    pytest.param(
        "centerpoint/detection_nuscenes_train.py",
        (*_SMOKE_RECIPE, "--model", "centerpoint-voxel01.nuscenes.mmdet3d", "--split", "mini", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="centerpoint/nuscenes-train",
    ),
    pytest.param(
        "centerpoint/detection_nuscenes_train.py",
        (*_SMOKE_RECIPE, "--model", "centerpoint-pillar02.nuscenes.mmdet3d", "--split", "mini", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="centerpoint-pillars/nuscenes-train",
    ),
    pytest.param(
        "transfusion/detection_nuscenes_train.py",
        (*_SMOKE_RECIPE, "--split", "mini", "--batch-size", "2"),
        marks=_GPU_SPCONV,
        id="transfusion/nuscenes-train",
    ),
    pytest.param(
        "lion/detection_nuscenes_train.py",
        (*_SMOKE_RECIPE, "--split", "mini", "--batch-size", "2"),
        marks=_GPU_MAMBA,
        id="lion/nuscenes-train",
    ),
]


@pytest.fixture(scope="module")
def examples_datasets_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One writable copy of the dummy datasets for the whole module, so dataset caches never touch the fixtures."""
    dest = tmp_path_factory.mktemp("datasets")
    shutil.copytree(DATASETS_DIR, dest, dirs_exist_ok=True)
    yield dest
    shutil.rmtree(dest, ignore_errors=True)


def _run(script: str, args: Tuple[str, ...], root: Path) -> None:
    command = [sys.executable, str(EXAMPLES_DIR / script), "--root", str(root), "--num-workers", "0", *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, f"{' '.join(command)}\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"


def _import(script: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(script, EXAMPLES_DIR / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_script_has_a_row() -> None:
    """Every example script is exercised by at least one row, or is named here with the reason it is not."""
    without_dummy_data = {
        "dgcnn/classification_modelnet40_benchmark.py",
        "octformer/segmentation_scannet200_benchmark.py",
        "point_transformer_v3/segmentation_s3dis_benchmark.py",
        "point_transformer_v3/segmentation_scannet200_benchmark.py",
        "point2vec/classification_modelnet40_benchmark.py",
        "pointgpt/classification_modelnet40_benchmark.py",
        "pointmlp/classification_modelnet40_benchmark.py",
        "pointnext/classification_modelnet40_benchmark.py",
        "sphereformer/segmentation_semantickitti_benchmark.py",
    }
    covered = {str(param.values[0]) for param in BENCHMARKS + TRAININGS}
    scripts = {path.relative_to(EXAMPLES_DIR).as_posix() for path in EXAMPLES_DIR.glob("*/*.py")}
    assert scripts - covered == without_dummy_data


@pytest.mark.example
@pytest.mark.pretrained
@pytest.mark.parametrize("script, args", BENCHMARKS)
def test_benchmark_script(script: str, args: Tuple[str, ...], examples_datasets_dir: Path) -> None:
    _run(script, args, examples_datasets_dir)


@pytest.mark.example
@pytest.mark.parametrize("script, args", TRAININGS)
def test_training_script(script: str, args: Tuple[str, ...], examples_datasets_dir: Path) -> None:
    _run(script, args, examples_datasets_dir)


def test_kitti_training_scripts_cache_the_objects_of_their_own_classes(tmp_path: Path) -> None:
    alone, after_cars = tmp_path / "alone", tmp_path / "after_cars"
    shutil.copytree(DATASETS_DIR / "KITTI", alone / "KITTI")
    shutil.copytree(DATASETS_DIR / "KITTI", after_cars / "KITTI")
    second = _import("second/detection_kitti_train.py")
    expected = second.configure_objects(Namespace(root=str(alone), train_split_file=None, force_process=False))
    args = Namespace(root=str(after_cars), train_split_file=None, force_process=False)
    _import("voxel_rcnn/detection_kitti_train.py").configure_objects(args)
    objects = second.configure_objects(args)
    assert len(expected) > 0
    assert [int(obj[DataKeys.LABEL]) for obj in objects] == [int(obj[DataKeys.LABEL]) for obj in expected]


class FirstInput(nn.Module):
    """Returns its first input, so a batch carries the logits a script scores."""

    def forward(self, x: Tensor, *args: Tensor) -> Tensor:
        return x


@pytest.mark.parametrize(
    "script, args",
    [
        pytest.param(
            "pointnet/classification_modelnet40_train.py",
            ("--batch-size", "2", "--val-batch-size", "2"),
            id="pointnet/modelnet40",
        ),
        pytest.param("pointnet/segmentation_s3dis_train.py", ("--batch-size", "2"), id="pointnet/s3dis"),
        pytest.param(
            "pvcnn/part_segmentation_shapenetpart_train.py",
            ("--batch-size", "2"),
            marks=_CLUSTER,
            id="pvcnn/shapenetpart",
        ),
    ],
)
def test_training_script_keeps_the_checkpoint_of_the_best_validation_score(
    script: str,
    args: Tuple[str, ...],
    examples_datasets_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import(script)
    scores = {2: 0.9, 4: 0.5, 5: 0.7}
    epochs: List[int] = []

    def train_one_epoch(model: nn.Module, *args: Any, **kwargs: Any) -> float:
        epochs.append(len(epochs) + 1)
        for tensor in model.state_dict().values():
            tensor.fill_(epochs[-1])
        return 1.0 / epochs[-1]

    def evaluate(*args: Any, **kwargs: Any) -> Dict[str, float]:
        score = scores[epochs[-1]]
        return {"acc": score, "mean_acc": score, "mIoU": score, "ins_mIoU": score}

    monkeypatch.setattr(module, "train_one_epoch", train_one_epoch)
    monkeypatch.setattr(module, "evaluate", evaluate)
    recipe = ("--limit-train-batches", "1", "--limit-val-batches", "1", "--epochs", "5", "--eval-every", "2")
    root = ("--root", str(examples_datasets_dir), "--num-workers", "0", "--device", "cpu")
    monkeypatch.setattr(sys, "argv", [script, *root, *recipe, "--output", str(tmp_path), *args])
    module.main()

    best = torch.load(tmp_path / "best.pt")
    last = torch.load(tmp_path / "last.pt")
    assert {float(tensor.flatten()[0]) for tensor in best.values()} == {2.0}
    assert {float(tensor.flatten()[0]) for tensor in last.values()} == {5.0}


def test_training_scripts_save_best_checkpoint_on_a_validation_score() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*/*_train.py")):
        tree = ast.parse(path.read_text())
        conditions = [
            ast.unparse(node.test)
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and "best.pt" in ast.unparse(node) and "best_" in ast.unparse(node.test)
        ]
        assert all("val[" in condition for condition in conditions), (path.name, conditions)
        assert "best.pt" not in path.read_text() or conditions, path.name


@pytest.mark.parametrize(
    "script",
    [
        pytest.param("spvcnn/segmentation_semantickitti_train.py", id="spvcnn"),
        pytest.param("sphereformer/segmentation_semantickitti_train.py", id="sphereformer"),
    ],
)
def test_semantickitti_validation_scores_each_batched_scan_with_its_own_voxels(script: str) -> None:
    module = _import(script)
    voxel_labels = [torch.tensor([0, 1]), torch.tensor([2, 3, 4])]
    inverses = [torch.tensor([0, 1, 1]), torch.tensor([2, 0, 1, 1])]
    scans: List[Dict[str, Any]] = [
        {
            DataKeys.X: nn.functional.one_hot(labels, module.NUM_CLASSES).float(),
            DataKeys.POS: torch.zeros(len(labels), 3),
            DataKeys.POS_GRID: torch.zeros(len(labels), 3, dtype=torch.long),
            DataKeys.INVERSE: inverse,
            DataKeys.ORIGIN_SEGMENT: labels[inverse],
        }
        for labels, inverse in zip(voxel_labels, inverses)
    ]
    batch = collate(scans, cat_keys=[DataKeys.INVERSE])
    metrics = module.evaluate(FirstInput(), [batch], "cpu")
    assert metrics["OA"] == 1.0


@pytest.mark.parametrize(
    "script",
    [
        pytest.param("spvcnn/segmentation_semantickitti_train.py", id="spvcnn"),
        pytest.param("sphereformer/segmentation_semantickitti_train.py", id="sphereformer"),
    ],
)
def test_semantickitti_validation_loader_indexes_the_inverse_map_per_scan(
    script: str, examples_datasets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _import(script)
    sequences = ("--train-sequences", "00", "--val-sequences", "08")
    monkeypatch.setattr(sys, "argv", [script, "--root", str(examples_datasets_dir), "--num-workers", "0", *sequences])
    _, val_loader = module.configure_dataloaders(module.parse_args())
    assert f"batch_{DataKeys.INVERSE}" in next(iter(val_loader))


def test_point_bert_scanobjectnn_validation_counts_a_prediction_past_the_classes_as_an_error() -> None:
    module = _import("point_bert/classification_scanobjectnn_train.py")

    class PastTheClasses(nn.Module):
        def forward(self, x: None, pos: Tensor, batch: Tensor) -> Tensor:
            return nn.functional.one_hot(torch.tensor([3, 20]), 40).float()

    batch = {
        DataKeys.POS: torch.zeros(2, 3),
        DataKeys.BATCH: torch.tensor([0, 1]),
        DataKeys.LABEL: torch.tensor([3, 5]),
    }
    metrics = module.evaluate(PastTheClasses(), [batch], "cpu")
    assert metrics["acc"] == 0.5


@pytest.mark.parametrize(
    "script",
    [
        pytest.param("kpconv/segmentation_semantickitti_benchmark.py", id="kpconv"),
        pytest.param("minkunet/segmentation_semantickitti_benchmark.py", id="minkunet"),
        pytest.param("randlanet/segmentation_semantickitti_benchmark.py", id="randlanet"),
        pytest.param("sphereformer/segmentation_semantickitti_benchmark.py", id="sphereformer"),
        pytest.param("spvcnn/segmentation_semantickitti_benchmark.py", marks=_GPU_TORCHSPARSE, id="spvcnn"),
    ],
)
def test_semantickitti_benchmark_rejects_the_unlabeled_test_split(script: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import(script)
    monkeypatch.setattr(sys, "argv", [script, "--split", "test"])
    with pytest.raises(SystemExit):
        module.parse_args()


def test_randlanet_semantickitti_training_merges_bus_on_rails_and_lane_marking() -> None:
    module = _import("randlanet/segmentation_semantickitti_train.py")
    raw = torch.tensor([13, 16, 256, 257, 60, 20, 40, 0])
    for transform in (module.TRAIN_TRANSFORM, module.VAL_TRANSFORM):
        relabel = transform.transforms[1]
        segment = relabel({DataKeys.SEGMENT: raw.clone()})[DataKeys.SEGMENT]
        assert segment.tolist() == [4, 4, 4, 4, 8, 4, 8, 255]


def test_kpconv_s3dis_training_decays_every_parameter_group(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import("kpconv/segmentation_s3dis_train.py")
    monkeypatch.setattr(sys, "argv", ["kpconv/segmentation_s3dis_train.py"])
    model = nn.ModuleDict({"conv": nn.Linear(2, 2), "offset": nn.Linear(2, 2)})
    optimizer, _ = module.configure_optimizers(module.parse_args(), model)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-3, 1e-3]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-2, 1e-3]


def test_readme_training_command_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    readme = (EXAMPLES_DIR / "README.md").read_text()
    commands = [line.split() for line in readme.splitlines() if line.startswith("uv run --no-sync python examples/")]
    assert len(commands) == 2
    for command in commands:
        script = command[4].removeprefix("examples/")
        monkeypatch.setattr(sys, "argv", [script, *command[5:]])
        _import(script).parse_args()


@pytest.mark.parametrize(
    ("script", "model", "expected"),
    [
        pytest.param(
            "pointnext/segmentation_s3dis_benchmark.py",
            "pointnext-xl.s3dis-area2.openpoints",
            ["Area_2"],
            id="pointnext-area2",
        ),
        pytest.param(
            "pointnext/segmentation_s3dis_benchmark.py",
            "pointnext-sm.s3dis-area5.openpoints",
            ["Area_5"],
            id="pointnext-area5",
        ),
        pytest.param(
            "pointnet2/segmentation_s3dis_benchmark.py",
            "pointnet2.s3dis-area3.openpoints",
            ["Area_3"],
            id="pointnet2-area3",
        ),
    ],
)
def test_s3dis_benchmark_defaults_to_the_held_out_area_of_the_checkpoint(
    script: str,
    model: str,
    expected: List[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import(script)
    monkeypatch.setattr(sys, "argv", [script, "--model", model])
    assert module.parse_args().areas == expected
