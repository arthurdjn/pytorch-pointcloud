# Third-Party Notices

`torch-pointcloud` is Apache-2.0 (see `LICENSE`). This file lists the third-party code, weights and
datasets used by the package and their licenses.

## Non-commercial material

Weights under a non-commercial license.

| material | source | license |
| --- | --- | --- |
| 5 Concerto checkpoints | Pointcept/Concerto | CC BY-NC 4.0 |
| 2 Utonia checkpoints | Pointcept/Utonia | CC BY-NC 4.0 |
| 2 Sonata checkpoints | facebookresearch/sonata | CC BY-NC 4.0 |

Sonata, Concerto and Utonia code is Apache-2.0; their weights are CC BY-NC 4.0.

## Pretrained weights

Checkpoints in the model registry, grouped by the release they were converted from. Each registry
entry records its license in the `license` field.

| source | author tag | license | count |
| --- | --- | --- | --- |
| guochengqian/PointNeXt, guochengqian/openpoints | `openpoints` | MIT | 36 |
| CGuangyan-BIT/PointGPT | `guangyan-chen` | MIT | 18 |
| antao97/dgcnn.pytorch | `an-tao` | MIT | 10 |
| Julie-tang00/Point-BERT | `xumin-yu` | MIT | 8 |
| Pang-Yatian/Point-MAE | `yatian-pang` | MIT | 7 |
| open-mmlab/OpenPCDet | `openpcdet` | Apache-2.0 | 6 |
| Pointcept/Concerto | `pointcept` | CC BY-NC 4.0 | 5 |
| ZrrSkywalker/Point-M2AE | `renrui-zhang` | MIT | 5 |
| LMD0311/PointMamba | `dingkang-liang` | Apache-2.0 | 5 |
| HuguesTHOMAS/KPConv-PyTorch | `hugues-thomas` | MIT | 4 |
| mit-han-lab/pvcnn, mit-han-lab/spvnas | `mit-han-lab` | MIT | 4 |
| ma-xu/pointMLP-pytorch | `xu-ma` | Apache-2.0 | 4 |
| Pointcept/Pointcept | `pointcept` | MIT | 4 |
| octree-nn/octformer | `octree-nn` | MIT | 3 |
| yanx27/Pointnet_Pointnet2_pytorch | `xu-yan` | MIT | 3 |
| facebookresearch/3detr | `fair` | Apache-2.0 | 3 |
| Pointcept/Utonia | `pointcept` | CC BY-NC 4.0 | 2 |
| facebookresearch/sonata | `fair` | CC BY-NC 4.0 | 2 |
| facebookresearch/votenet | `fair` | MIT | 2 |
| DylanWusee/pointconv_pytorch | `wenxuan-wu` | MIT | 1 |
| tsunghan-wu/RandLA-Net-pytorch | `tsung-han-wu` | MIT | 1 |
| happinesslz/LION | `zhe-liu` | Apache-2.0 | 1 |

## Training data

Licenses of the datasets the checkpoints were trained on.

| dataset | terms | checkpoints |
| --- | --- | --- |
| S3DIS (Stanford 2D-3D-S) | academic research agreement | 43 |
| ScanObjectNN | terms of use, research only | 24 |
| ModelNet40 | academic research only | 23 |
| ShapeNet, ShapeNetPart | non-commercial research and educational use | 14 |
| ScanNet | terms of use, non-commercial research and educational use | 13 |
| SemanticKITTI | CC BY-NC-SA 4.0 | 4 |
| nuScenes | CC BY-NC-SA 4.0 | 4 |
| KITTI | CC BY-NC-SA 3.0 | 3 |
| SUN RGB-D | research use | 2 |
| HM3D, ArkitScenes (self-supervised pretraining) | non-commercial | 6 |

## Source code

Files adapted from other implementations. Paths are relative to `src/torch_pointcloud/`.

| source | license | files |
| --- | --- | --- |
| open-mmlab/OpenPCDet | Apache-2.0 | `models/pointpillars.py`, `models/pointrcnn.py`, `models/second.py`, `models/voxelnext.py`, `models/voxel_mamba.py` |
| facebookresearch/3detr | Apache-2.0 | `models/detr3d.py` |
| facebookresearch/votenet | MIT | `models/votenet.py`, `lightning/callbacks.py` |
| Pointcept/Pointcept | MIT | `models/point_transformer_v3.py` |
| pyg-team/pytorch_geometric | MIT | `models/point_transformer.py` |
| Project-MONAI/MONAI | Apache-2.0 | `datasets/utils.py` |
| mit-han-lab/pvcnn | MIT | `utils/voxelization.py` |
| mit-han-lab/bevfusion | Apache-2.0 | `models/lion.py` |
| happinesslz/LION | Apache-2.0 | `models/lion.py` |
| gwenzhang/Voxel-Mamba | Apache-2.0 | `models/voxel_mamba.py`, `layers/vfe.py` |
| JIA-Lab-research/VoxelNeXt | Apache-2.0 | `models/voxelnext.py` |
| sunjiahao1999/SPFormer | MIT | `models/spformer_unet.py` |
| Julie-tang00/Point-BERT | MIT | `models/point_bert.py` |
| Pang-Yatian/Point-MAE | MIT | `models/point_mae.py` |
| ZrrSkywalker/Point-M2AE | MIT | `models/point_m2ae.py` |
| CGuangyan-BIT/PointGPT | MIT | `models/pointgpt.py` |
| LMD0311/PointMamba | Apache-2.0 | `models/point_mamba.py` |
