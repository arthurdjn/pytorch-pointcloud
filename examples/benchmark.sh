#!/usr/bin/env bash
# Benchmark every pretrained checkpoint of the model zoo, one line per checkpoint, fastest first.
#
# Usage:
#     ROOT=/path/to/datasets bash examples/benchmark.sh

set -euo pipefail

ROOT="${ROOT:-data}"
RUN="uv run --no-sync python"

# Classification (ModelNet40, ScanObjectNN)
$RUN examples/pointnet2_benchmark_classification.py --root "$ROOT" --model pointnet2-ssg.modelnet40.xu-yan
$RUN examples/pointnet2_benchmark_classification.py --root "$ROOT" --model pointnet2-msg.modelnet40.xu-yan
$RUN examples/pointnet2_benchmark_classification.py --root "$ROOT" --model pointnet2.modelnet40.openpoints
$RUN examples/pointnet2_benchmark_classification.py --root "$ROOT" --model pointnet2.scanobjectnn-hardest.openpoints
$RUN examples/pointnext_benchmark_classification.py --root "$ROOT" --model pointnext-sm-c64.modelnet40.openpoints
$RUN examples/pointnext_benchmark_classification.py --root "$ROOT" --model pointnext-sm.scanobjectnn-hardest.openpoints
$RUN examples/pointmlp_benchmark_classification.py --root "$ROOT" --model pointmlp-base.modelnet40.xu-ma
$RUN examples/pointmlp_benchmark_classification.py --root "$ROOT" --model pointmlp-elite.modelnet40.xu-ma
$RUN examples/pointmlp_benchmark_classification.py --root "$ROOT" --model pointmlp-base.scanobjectnn-hardest.xu-ma
$RUN examples/pointmlp_benchmark_classification.py --root "$ROOT" --model pointmlp-elite.scanobjectnn-hardest.xu-ma
$RUN examples/pointconv_benchmark_classification.py --root "$ROOT" --model pointconv-density-base.modelnet40.wenxuan-wu
$RUN examples/dgcnn_benchmark_classification.py --root "$ROOT" --model dgcnn.modelnet40-1024.an-tao
$RUN examples/dgcnn_benchmark_classification.py --root "$ROOT" --model dgcnn.modelnet40-2048.an-tao
$RUN examples/point_mae_benchmark_classification.py --root "$ROOT" --model point-mae-base.modelnet40.yatian-pang
$RUN examples/point_mae_benchmark_classification.py --root "$ROOT" --model point-mae-base.modelnet40-8k.yatian-pang
$RUN examples/point_mae_benchmark_classification.py --root "$ROOT" --model point-mae-base.scanobjectnn-objbg.yatian-pang
$RUN examples/point_mae_benchmark_classification.py --root "$ROOT" --model point-mae-base.scanobjectnn-objonly.yatian-pang
$RUN examples/point_mae_benchmark_classification.py --root "$ROOT" --model point-mae-base.scanobjectnn-hardest.yatian-pang
$RUN examples/point_bert_benchmark_classification.py --root "$ROOT" --model point-bert-base.modelnet40.xumin-yu
$RUN examples/point_bert_benchmark_classification.py --root "$ROOT" --model point-bert-base.modelnet40-4k.xumin-yu
$RUN examples/point_bert_benchmark_classification.py --root "$ROOT" --model point-bert-base.modelnet40-8k.xumin-yu
$RUN examples/point_bert_benchmark_classification.py --root "$ROOT" --model point-bert-base.scanobjectnn-objbg.xumin-yu
$RUN examples/point_bert_benchmark_classification.py --root "$ROOT" --model point-bert-base.scanobjectnn-objonly.xumin-yu
$RUN examples/point_bert_benchmark_classification.py --root "$ROOT" --model point-bert-base.scanobjectnn-hardest.xumin-yu
$RUN examples/point_m2ae_benchmark_classification.py --root "$ROOT" --model point-m2ae-base.modelnet40.renrui-zhang
$RUN examples/point_m2ae_benchmark_classification.py --root "$ROOT" --model point-m2ae-base.scanobjectnn-objbg.renrui-zhang
$RUN examples/point_m2ae_benchmark_classification.py --root "$ROOT" --model point-m2ae-base.scanobjectnn-hardest.renrui-zhang
$RUN examples/point_mamba_benchmark_classification.py --root "$ROOT" --model point-mamba-base.modelnet40.dingkang-liang
$RUN examples/point_mamba_benchmark_classification.py --root "$ROOT" --model point-mamba-base.scanobjectnn-objbg.dingkang-liang
$RUN examples/point_mamba_benchmark_classification.py --root "$ROOT" --model point-mamba-base.scanobjectnn-objonly.dingkang-liang
$RUN examples/point_mamba_benchmark_classification.py --root "$ROOT" --model point-mamba-base.scanobjectnn-hardest.dingkang-liang
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-s.modelnet40.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-b.modelnet40.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-l.modelnet40.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-s.modelnet40-8k.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-b.modelnet40-8k.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-l.modelnet40-8k.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-s.scanobjectnn-objbg.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-b.scanobjectnn-objbg.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-l.scanobjectnn-objbg.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-s.scanobjectnn-objonly.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-b.scanobjectnn-objonly.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-l.scanobjectnn-objonly.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-s.scanobjectnn-hardest.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-b.scanobjectnn-hardest.guangyan-chen
$RUN examples/pointgpt_benchmark_classification.py --root "$ROOT" --model pointgpt-l.scanobjectnn-hardest.guangyan-chen
$RUN examples/octformer_benchmark_classification.py --root "$ROOT" --model octformer-base.modelnet40.octree-nn

# Part segmentation (ShapeNetPart)
$RUN examples/dgcnn_benchmark_part_segmentation.py --root "$ROOT" --model dgcnn.shapenetpart.an-tao
$RUN examples/pointnext_benchmark_part_segmentation.py --root "$ROOT" --model pointnext-sm.shapenetpart.openpoints
$RUN examples/pointnext_benchmark_part_segmentation.py --root "$ROOT" --model pointnext-sm-c64.shapenetpart.openpoints
$RUN examples/pointnext_benchmark_part_segmentation.py --root "$ROOT" --model pointnext-sm-c160.shapenetpart.openpoints
$RUN examples/point_mae_benchmark_part_segmentation.py --root "$ROOT" --model point-mae-base.shapenetpart.yatian-pang
$RUN examples/point_m2ae_benchmark_part_segmentation.py --root "$ROOT" --model point-m2ae-base.shapenetpart.renrui-zhang

# Indoor detection (SUN RGB-D, ScanNet)
$RUN examples/votenet_benchmark_detection.py --root "$ROOT" --model votenet.sunrgbd.fair
$RUN examples/votenet_benchmark_detection.py --root "$ROOT" --model votenet.scannet.fair
$RUN examples/3detr_benchmark_detection.py --root "$ROOT" --model 3detr.sunrgbd.fair
$RUN examples/3detr_benchmark_detection.py --root "$ROOT" --model 3detr.scannet.fair
$RUN examples/3detr_benchmark_detection.py --root "$ROOT" --model 3detr-m.scannet.fair

# Outdoor detection (KITTI, nuScenes)
$RUN examples/pointpillars_benchmark_detection.py --root "$ROOT" --model pointpillars.kitti.openpcdet --split-file "$ROOT/KITTI/raw/ImageSets/val.txt"
$RUN examples/pointpillars_benchmark_detection.py --root "$ROOT" --model pointpillars-multihead.nuscenes.openpcdet
$RUN examples/second_benchmark_detection.py --root "$ROOT" --model second.kitti.openpcdet --split-file "$ROOT/KITTI/raw/ImageSets/val.txt"
$RUN examples/second_benchmark_detection.py --root "$ROOT" --model second-multihead.nuscenes.openpcdet
$RUN examples/pointrcnn_benchmark_detection.py --root "$ROOT" --model pointrcnn.kitti.openpcdet --split-file "$ROOT/KITTI/raw/ImageSets/val.txt"
$RUN examples/voxelnext_benchmark_detection.py --root "$ROOT" --model voxelnext.nuscenes.openpcdet
$RUN examples/lion_benchmark_detection.py --root "$ROOT" --model lion-mamba.nuscenes.zhe-liu

# Segmentation (S3DIS, ScanNet, SemanticKITTI)
$RUN examples/dgcnn_benchmark_segmentation.py --root "$ROOT" --dataset scannet
$RUN examples/dgcnn_benchmark_segmentation.py --root "$ROOT" --dataset s3dis --area 1
$RUN examples/dgcnn_benchmark_segmentation.py --root "$ROOT" --dataset s3dis --area 2
$RUN examples/dgcnn_benchmark_segmentation.py --root "$ROOT" --dataset s3dis --area 3
$RUN examples/dgcnn_benchmark_segmentation.py --root "$ROOT" --dataset s3dis --area 4
$RUN examples/dgcnn_benchmark_segmentation.py --root "$ROOT" --dataset s3dis --area 5
$RUN examples/dgcnn_benchmark_segmentation.py --root "$ROOT" --dataset s3dis --area 6
$RUN examples/pvcnn_benchmark_segmentation.py --root "$ROOT" --model pvcnn.s3dis-area5.mit-han-lab --areas Area_5
$RUN examples/kpconv_benchmark_segmentation.py --root "$ROOT" --model kpfcnn-base.s3dis-area5.hugues-thomas
$RUN examples/kpconv_benchmark_segmentation.py --root "$ROOT" --model kpfcnn-base-sm.s3dis-area5.hugues-thomas
$RUN examples/kpconv_benchmark_segmentation.py --root "$ROOT" --model kpfcnn-base-deform.s3dis-area5.hugues-thomas
$RUN examples/kpconv_benchmark_segmentation.py --root "$ROOT" --model kpfcnn-base-sm-deform.s3dis-area5.hugues-thomas
$RUN examples/pointnet2_benchmark_segmentation.py --root "$ROOT" --model pointnet2.s3dis-area5.xu-yan --areas Area_5
$RUN examples/pointnet2_benchmark_segmentation.py --root "$ROOT" --model pointnet2.s3dis-area1.openpoints --areas Area_1 --sub-batch-size 4
$RUN examples/pointnet2_benchmark_segmentation.py --root "$ROOT" --model pointnet2.s3dis-area2.openpoints --areas Area_2 --sub-batch-size 1
$RUN examples/pointnet2_benchmark_segmentation.py --root "$ROOT" --model pointnet2.s3dis-area3.openpoints --areas Area_3 --sub-batch-size 4
$RUN examples/pointnet2_benchmark_segmentation.py --root "$ROOT" --model pointnet2.s3dis-area4.openpoints --areas Area_4 --sub-batch-size 4
$RUN examples/pointnet2_benchmark_segmentation.py --root "$ROOT" --model pointnet2.s3dis-area5.openpoints --areas Area_5 --sub-batch-size 4
$RUN examples/pointnet2_benchmark_segmentation.py --root "$ROOT" --model pointnet2.s3dis-area6.openpoints --areas Area_6 --sub-batch-size 4
$RUN examples/randlanet_benchmark_segmentation.py --root "$ROOT" --model randlanet.semantickitti.tsung-han-wu
$RUN examples/spvcnn_benchmark_segmentation.py --root "$ROOT" --model spvcnn-119gmacs.semantickitti.mit-han-lab
$RUN examples/spvcnn_benchmark_segmentation.py --root "$ROOT" --model spvcnn-47gmacs.semantickitti.mit-han-lab
$RUN examples/spvcnn_benchmark_segmentation.py --root "$ROOT" --model spvcnn-30gmacs.semantickitti.mit-han-lab
$RUN examples/octformer_benchmark_segmentation.py --root "$ROOT" --model octformer-base.scannet20.octree-nn
$RUN examples/octformer_benchmark_segmentation.py --root "$ROOT" --model octformer-base.scannet200.octree-nn
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-sm.s3dis-area1.openpoints --areas Area_1 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-sm.s3dis-area2.openpoints --areas Area_2 --sub-batch-size 1
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-sm.s3dis-area3.openpoints --areas Area_3 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-sm.s3dis-area4.openpoints --areas Area_4 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-sm.s3dis-area5.openpoints --areas Area_5 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-sm.s3dis-area6.openpoints --areas Area_6 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-base.s3dis-area1.openpoints --areas Area_1 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-base.s3dis-area2.openpoints --areas Area_2 --sub-batch-size 1
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-base.s3dis-area3.openpoints --areas Area_3 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-base.s3dis-area4.openpoints --areas Area_4 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-base.s3dis-area5.openpoints --areas Area_5 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-base.s3dis-area6.openpoints --areas Area_6 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-lg.s3dis-area1.openpoints --areas Area_1 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-lg.s3dis-area2.openpoints --areas Area_2 --sub-batch-size 1
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-lg.s3dis-area3.openpoints --areas Area_3 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-lg.s3dis-area4.openpoints --areas Area_4 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-lg.s3dis-area5.openpoints --areas Area_5 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-lg.s3dis-area6.openpoints --areas Area_6 --sub-batch-size 8
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-xl.s3dis-area1.openpoints --areas Area_1 --sub-batch-size 2
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-xl.s3dis-area2.openpoints --areas Area_2 --sub-batch-size 1
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-xl.s3dis-area3.openpoints --areas Area_3 --sub-batch-size 2
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-xl.s3dis-area4.openpoints --areas Area_4 --sub-batch-size 2
$RUN examples/pointnext_benchmark_segmentation.py --root "$ROOT" --model pointnext-xl.s3dis-area5.openpoints --areas Area_5 --sub-batch-size 2

# Segmentation with TTA (ScanNet, S3DIS)
$RUN examples/spunet_benchmark_segmentation.py --root "$ROOT" --model spunet-v1m1.scannet20.pointcept --sub-batch-size 8
$RUN examples/ptv3_benchmark_segmentation.py --root "$ROOT" --model ptv3-base.scannet20.pointcept --sub-batch-size 8
$RUN examples/ptv3_benchmark_segmentation.py --root "$ROOT" --model ptv3-base.scannet200.pointcept --sub-batch-size 8
$RUN examples/ptv3_benchmark_segmentation.py --root "$ROOT" --model ptv3-base.s3dis-area5.pointcept --sub-batch-size 1
$RUN examples/sonata_benchmark_segmentation.py --root "$ROOT" --model sonata-lp.scannet20.fair --sub-batch-size 4
$RUN examples/concerto_benchmark_segmentation.py --root "$ROOT" --model concerto-large-lp.scannet20.pointcept --sub-batch-size 2
$RUN examples/utonia_benchmark_segmentation.py --root "$ROOT" --model utonia-lp.scannet20.pointcept --sub-batch-size 2
