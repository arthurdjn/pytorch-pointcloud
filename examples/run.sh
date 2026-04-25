#!/bin/bash

# DGCNN Classification
uv run --no-sync python examples/dgcnn_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --model dgcnn-base \
    --dataset modelnet10

uv run --no-sync python examples/dgcnn_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --model dgcnn-base \
    --dataset modelnet40

# DGCNN Segmentation
uv run --no-sync python examples/dgcnn_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --model dgcnn-base \
    --dataset shapenetpart

uv run --no-sync python examples/dgcnn_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --model dgcnn-base \
    --dataset s3dis

# DGCNN Benchmark
uv run --no-sync python examples/dgcnn_benchmark_modelnet.py --model dgcnn-antao.modelnet40.1024

uv run --no-sync python examples/dgcnn_benchmark_modelnet.py --model dgcnn-antao.modelnet40.2048

uv run --no-sync python examples/dgcnn_benchmark_shapenetpart.py --model dgcnn-antao.shapenetpart

uv run --no-sync python examples/dgcnn_benchmark_s3dis.py --area 1
uv run --no-sync python examples/dgcnn_benchmark_s3dis.py --area 2
uv run --no-sync python examples/dgcnn_benchmark_s3dis.py --area 3
uv run --no-sync python examples/dgcnn_benchmark_s3dis.py --area 4
uv run --no-sync python examples/dgcnn_benchmark_s3dis.py --area 5
uv run --no-sync python examples/dgcnn_benchmark_s3dis.py --area 6

uv run --no-sync python examples/dgcnn_scannet_benchmark_antao.py
uv run --no-sync python examples/dgcnn_benchmark_scannet.py


# KPConv Classification
uv run --no-sync python examples/kpconv_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/kpconv_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# KPConv Segmentation
uv run --no-sync python examples/kpconv_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/kpconv_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis

# KPConv Benchmark
uv run --no-sync python examples/kpconv_benchmark_s3dis.py --model kpfcnn-base.s3dis

uv run --no-sync python examples/kpconv_benchmark_s3dis.py --model kpfcnn-base-sm.s3dis

uv run --no-sync python examples/kpconv_benchmark_s3dis.py --model kpfcnn-base-deform.s3dis

uv run --no-sync python examples/kpconv_benchmark_s3dis.py --model kpfcnn-base-sm-deform.s3dis


# OctFormer Classification
uv run --no-sync python examples/octformer_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/octformer_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# OctFormer Segmentation
uv run --no-sync python examples/octformer_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/octformer_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis

# OctFormer benchmark
uv run --no-sync python examples/octformer_benchmark_modelnet.py --model octformer-base.modelnet40

uv run --no-sync python examples/octformer_benchmark_scannet.py --model octformer-base.scannet20

uv run --no-sync python examples/octformer_benchmark_scannet.py --model octformer-base.scannet200


# PointMamba Classification
uv run --no-sync python examples/point_mamba_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/point_mamba_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointMamba Benchmark
uv run --no-sync python examples/point_mamba_benchmark_modelnet.py --model point-mamba-base.modelnet40

uv run --no-sync python examples/point_mamba_benchmark_scanobjectnn.py --model point-mamba-base.scanobjectnn

uv run --no-sync python examples/point_mamba_benchmark_scanobjectnn.py --model point-mamba-base.scanobjectnn-nobg

uv run --no-sync python examples/point_mamba_benchmark_scanobjectnn.py --model point-mamba-base.scanobjectnn-augmentedrot-scale75


# PointTransformer Classification
uv run --no-sync python examples/point_transformer_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/point_transformer_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointTransformer Segmentation
uv run --no-sync python examples/point_transformer_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/point_transformer_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# PointTransformerV2 Classification
uv run --no-sync python examples/point_transformer_v2_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/point_transformer_v2_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointTransformerV2 Segmentation
uv run --no-sync python examples/point_transformer_v2_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/point_transformer_v2_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# PointTransformerV3 Classification
uv run --no-sync python examples/point_transformer_v3_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/point_transformer_v3_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointTransformerV3 Segmentation
uv run --no-sync python examples/point_transformer_v3_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/point_transformer_v3_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# PointCNN Classification
uv run --no-sync python examples/pointcnn_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/pointcnn_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointCNN Segmentation
uv run --no-sync python examples/pointcnn_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/pointcnn_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# PointConv Classification
uv run --no-sync python examples/pointconv_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/pointconv_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40


# PointMLP Classification
uv run --no-sync python examples/pointmlp_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/pointmlp_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointMLP Segmentation
uv run --no-sync python examples/pointmlp_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/pointmlp_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# PointNet Classification
uv run --no-sync python examples/pointnet_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/pointnet_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointNet Segmentation
uv run --no-sync python examples/pointnet_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/pointnet_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# PointNet2 Classification
uv run --no-sync python examples/pointnet2_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/pointnet2_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointNet2 Segmentation
uv run --no-sync python examples/pointnet2_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/pointnet2_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# PointNeXt Classification
uv run --no-sync python examples/pointnext_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/pointnext_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PointNeXt Segmentation
uv run --no-sync python examples/pointnext_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/pointnext_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis

# PointNeXt Benchmark - Classification
uv run --no-sync python examples/pointnext_benchmark_scanobjectnn.py --model pointnext-sm.scanobjectnn

uv run --no-sync python examples/pointnext_benchmark_modelnet.py --model pointnext-sm-c64.modelnet40

# PointNeXt Benchmark - ShapeNetPart
uv run --no-sync python examples/pointnext_benchmark_shapenetpart.py --model pointnext-sm.shapenetpart
uv run --no-sync python examples/pointnext_benchmark_shapenetpart.py --model pointnext-sm-c64.shapenetpart

# PointNeXt Benchmark - S3DIS (6-fold cross-validation, sm)
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-sm.s3dis-area1 --areas Area_1
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-sm.s3dis-area2 --areas Area_2
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-sm.s3dis-area3 --areas Area_3
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-sm.s3dis-area4 --areas Area_4
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-sm.s3dis-area5 --areas Area_5
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-sm.s3dis-area6 --areas Area_6

# PointNeXt Benchmark - S3DIS (6-fold cross-validation, base)
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-base.s3dis-area1 --areas Area_1
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-base.s3dis-area2 --areas Area_2
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-base.s3dis-area3 --areas Area_3
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-base.s3dis-area4 --areas Area_4
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-base.s3dis-area5 --areas Area_5
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-base.s3dis-area6 --areas Area_6

# PointNeXt Benchmark - S3DIS (6-fold cross-validation, lg)
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-lg.s3dis-area1 --areas Area_1
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-lg.s3dis-area2 --areas Area_2
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-lg.s3dis-area3 --areas Area_3
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-lg.s3dis-area4 --areas Area_4
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-lg.s3dis-area5 --areas Area_5
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-lg.s3dis-area6 --areas Area_6

# PointNeXt Benchmark - S3DIS (6-fold cross-validation, xl)
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-xl.s3dis-area1 --areas Area_1
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-xl.s3dis-area2 --areas Area_2
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-xl.s3dis-area3 --areas Area_3
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-xl.s3dis-area4 --areas Area_4
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-xl.s3dis-area5 --areas Area_5
uv run --no-sync python examples/pointnext_benchmark_s3dis.py --model pointnext-xl.s3dis-area6 --areas Area_6


# PVCNN Classification
uv run --no-sync python examples/pvcnn_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/pvcnn_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PVCNN Segmentation
uv run --no-sync python examples/pvcnn_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/pvcnn_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# PVCNN2 Classification
uv run --no-sync python examples/pvcnn2_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/pvcnn2_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# PVCNN2 Segmentation
uv run --no-sync python examples/pvcnn2_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/pvcnn2_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# RandLANet Classification
uv run --no-sync python examples/randlanet_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/randlanet_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# RandLANet Segmentation
uv run --no-sync python examples/randlanet_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/randlanet_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis


# SPVCNN Classification
uv run --no-sync python examples/spvcnn_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet10

uv run --no-sync python examples/spvcnn_classification.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset modelnet40

# SPVCNN Segmentation
uv run --no-sync python examples/spvcnn_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset shapenetpart

uv run --no-sync python examples/spvcnn_segmentation.py \
    --limit-train-batches 5 \
    --limit-test-batches 5 \
    --epochs 1 \
    --dataset s3dis
