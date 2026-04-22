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
