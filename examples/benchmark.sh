#!/usr/bin/env bash
# Benchmark every pretrained checkpoint of the model zoo, one line per checkpoint, fastest first.
#
# Usage:
#     ROOT=/path/to/datasets bash examples/benchmark.sh

set -euo pipefail

ROOT="${ROOT:-data}"
RUN="uv run --no-sync python"

# Classification (ModelNet40, ScanObjectNN)
$RUN examples/pointnet2/classification_scanobjectnn_benchmark.py --root "$ROOT" --model pointnet2.scanobjectnn-hardest.openpoints
$RUN examples/pointnext/classification_modelnet40_benchmark.py --root "$ROOT" --model pointnext-sm-c64.modelnet40.openpoints
$RUN examples/dgcnn/classification_modelnet40_benchmark.py --root "$ROOT" --model dgcnn.modelnet40-1024.an-tao
$RUN examples/dgcnn/classification_modelnet40_benchmark.py --root "$ROOT" --model dgcnn.modelnet40-2048.an-tao
$RUN examples/point_bert/classification_scanobjectnn_benchmark.py --root "$ROOT" --model point-bert-base.scanobjectnn-objbg.xumin-yu
$RUN examples/point_bert/classification_scanobjectnn_benchmark.py --root "$ROOT" --model point-bert-base.scanobjectnn-objonly.xumin-yu
$RUN examples/point_bert/classification_scanobjectnn_benchmark.py --root "$ROOT" --model point-bert-base.scanobjectnn-hardest.xumin-yu
$RUN examples/point_m2ae/classification_scanobjectnn_benchmark.py --root "$ROOT" --model point-m2ae-base.scanobjectnn-objbg.renrui-zhang
$RUN examples/point_m2ae/classification_scanobjectnn_benchmark.py --root "$ROOT" --model point-m2ae-base.scanobjectnn-hardest.renrui-zhang
$RUN examples/octformer/classification_modelnet40_benchmark.py --root "$ROOT" --model octformer-base.modelnet40.octree-nn

# Part segmentation (ShapeNetPart)
$RUN examples/dgcnn/part_segmentation_shapenetpart_benchmark.py --root "$ROOT" --model dgcnn.shapenetpart.an-tao
$RUN examples/pointnext/part_segmentation_shapenetpart_benchmark.py --root "$ROOT" --model pointnext-sm.shapenetpart.openpoints
$RUN examples/pointnext/part_segmentation_shapenetpart_benchmark.py --root "$ROOT" --model pointnext-sm-c64.shapenetpart.openpoints
$RUN examples/pointnext/part_segmentation_shapenetpart_benchmark.py --root "$ROOT" --model pointnext-sm-c160.shapenetpart.openpoints
$RUN examples/point_mae/part_segmentation_shapenetpart_benchmark.py --root "$ROOT" --model point-mae-base.shapenetpart.yatian-pang
$RUN examples/point_m2ae/part_segmentation_shapenetpart_benchmark.py --root "$ROOT" --model point-m2ae-base.shapenetpart.renrui-zhang

# Indoor detection (SUN RGB-D, ScanNet)
$RUN examples/votenet/detection_sunrgbd_benchmark.py --root "$ROOT" --model votenet.sunrgbd.fair
$RUN examples/threedetr/detection_scannet_benchmark.py --root "$ROOT" --model 3detr.scannet.fair
$RUN examples/threedetr/detection_scannet_benchmark.py --root "$ROOT" --model 3detr-m.scannet.fair

# Outdoor detection (KITTI, nuScenes)
$RUN examples/pointrcnn/detection_kitti_benchmark.py --root "$ROOT" --model pointrcnn.kitti.openpcdet --split-file "$ROOT/KITTI/raw/ImageSets/val.txt"
$RUN examples/voxelnext/detection_nuscenes_benchmark.py --root "$ROOT" --model voxelnext.nuscenes.openpcdet
$RUN examples/lion/detection_nuscenes_benchmark.py --root "$ROOT" --model lion-mamba.nuscenes.zhe-liu

# Segmentation (S3DIS, ScanNet, SemanticKITTI)
$RUN examples/dgcnn/segmentation_scannet_benchmark.py --root "$ROOT" --dataset scannet
$RUN examples/dgcnn/segmentation_scannet_benchmark.py --root "$ROOT" --dataset s3dis --area 1
$RUN examples/dgcnn/segmentation_scannet_benchmark.py --root "$ROOT" --dataset s3dis --area 2
$RUN examples/dgcnn/segmentation_scannet_benchmark.py --root "$ROOT" --dataset s3dis --area 3
$RUN examples/dgcnn/segmentation_scannet_benchmark.py --root "$ROOT" --dataset s3dis --area 4
$RUN examples/dgcnn/segmentation_scannet_benchmark.py --root "$ROOT" --dataset s3dis --area 5
$RUN examples/dgcnn/segmentation_scannet_benchmark.py --root "$ROOT" --dataset s3dis --area 6
$RUN examples/pvcnn/segmentation_s3dis_benchmark.py --root "$ROOT" --model pvcnn.s3dis-area5.mit-han-lab --areas Area_5
$RUN examples/kpconv/segmentation_s3dis_benchmark.py --root "$ROOT" --model kpfcnn-base.s3dis-area5.hugues-thomas
$RUN examples/kpconv/segmentation_s3dis_benchmark.py --root "$ROOT" --model kpfcnn-base-sm.s3dis-area5.hugues-thomas
$RUN examples/kpconv/segmentation_s3dis_benchmark.py --root "$ROOT" --model kpfcnn-base-deform.s3dis-area5.hugues-thomas
$RUN examples/kpconv/segmentation_s3dis_benchmark.py --root "$ROOT" --model kpfcnn-base-sm-deform.s3dis-area5.hugues-thomas
$RUN examples/pointnet2/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnet2.s3dis-area5.xu-yan --areas Area_5
$RUN examples/pointnet2/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnet2.s3dis-area1.openpoints --areas Area_1 --sw-batch-size 4
$RUN examples/pointnet2/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnet2.s3dis-area2.openpoints --areas Area_2 --sw-batch-size 1
$RUN examples/pointnet2/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnet2.s3dis-area3.openpoints --areas Area_3 --sw-batch-size 4
$RUN examples/pointnet2/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnet2.s3dis-area4.openpoints --areas Area_4 --sw-batch-size 4
$RUN examples/pointnet2/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnet2.s3dis-area5.openpoints --areas Area_5 --sw-batch-size 4
$RUN examples/pointnet2/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnet2.s3dis-area6.openpoints --areas Area_6 --sw-batch-size 4
$RUN examples/randlanet/segmentation_semantickitti_benchmark.py --root "$ROOT" --model randlanet.semantickitti.tsung-han-wu
$RUN examples/spvcnn/segmentation_semantickitti_benchmark.py --root "$ROOT" --model spvcnn-119gmacs.semantickitti.mit-han-lab
$RUN examples/spvcnn/segmentation_semantickitti_benchmark.py --root "$ROOT" --model spvcnn-47gmacs.semantickitti.mit-han-lab
$RUN examples/spvcnn/segmentation_semantickitti_benchmark.py --root "$ROOT" --model spvcnn-30gmacs.semantickitti.mit-han-lab
$RUN examples/octformer/segmentation_scannet_benchmark.py --root "$ROOT" --model octformer-base.scannet20.octree-nn
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-sm.s3dis-area1.openpoints --areas Area_1 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-sm.s3dis-area2.openpoints --areas Area_2 --sw-batch-size 1
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-sm.s3dis-area3.openpoints --areas Area_3 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-sm.s3dis-area4.openpoints --areas Area_4 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-sm.s3dis-area5.openpoints --areas Area_5 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-sm.s3dis-area6.openpoints --areas Area_6 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-base.s3dis-area1.openpoints --areas Area_1 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-base.s3dis-area2.openpoints --areas Area_2 --sw-batch-size 1
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-base.s3dis-area3.openpoints --areas Area_3 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-base.s3dis-area4.openpoints --areas Area_4 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-base.s3dis-area5.openpoints --areas Area_5 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-base.s3dis-area6.openpoints --areas Area_6 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-lg.s3dis-area1.openpoints --areas Area_1 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-lg.s3dis-area2.openpoints --areas Area_2 --sw-batch-size 1
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-lg.s3dis-area3.openpoints --areas Area_3 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-lg.s3dis-area4.openpoints --areas Area_4 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-lg.s3dis-area5.openpoints --areas Area_5 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-lg.s3dis-area6.openpoints --areas Area_6 --sw-batch-size 8
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-xl.s3dis-area1.openpoints --areas Area_1 --sw-batch-size 2
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-xl.s3dis-area2.openpoints --areas Area_2 --sw-batch-size 1
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-xl.s3dis-area3.openpoints --areas Area_3 --sw-batch-size 2
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-xl.s3dis-area4.openpoints --areas Area_4 --sw-batch-size 2
$RUN examples/pointnext/segmentation_s3dis_benchmark.py --root "$ROOT" --model pointnext-xl.s3dis-area5.openpoints --areas Area_5 --sw-batch-size 2

# Segmentation with TTA (ScanNet, S3DIS)
$RUN examples/spunet/segmentation_scannet_benchmark.py --root "$ROOT" --model spunet-v1m1.scannet20.pointcept --sw-batch-size 8
$RUN examples/sonata/segmentation_scannet_benchmark.py --root "$ROOT" --model sonata-lp.scannet20.fair --sw-batch-size 4
$RUN examples/concerto/segmentation_scannet_benchmark.py --root "$ROOT" --model concerto-large-lp.scannet20.pointcept --sw-batch-size 2
$RUN examples/utonia/segmentation_scannet_benchmark.py --root "$ROOT" --model utonia-lp.scannet20.pointcept --sw-batch-size 2
