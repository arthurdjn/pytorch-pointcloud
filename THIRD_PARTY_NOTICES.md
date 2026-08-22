# Third-Party Notices

`torch-pointcloud` is released under the Apache License 2.0 (see `LICENSE`). This file records the
third-party material the package adapts, redistributes, or downloads at runtime, and the terms that
material stays under.

Pretrained checkpoints are **not** covered by the project's Apache-2.0 grant. Each checkpoint keeps
the license of the release it was converted from, recorded in the `license` field of its registry
entry.

## Non-commercial material

The following may be used for research or evaluation only. It is not available for commercial use
under any terms this project can grant.

| material | source | license |
| --- | --- | --- |
| `models/oneformer3d.py`, 2 OneFormer3D checkpoints | filaPro/oneformer3d | CC BY-NC 4.0 |
| 7 Concerto and Utonia checkpoints | Pointcept/Pointcept | CC BY-NC 4.0 |
| 2 Sonata checkpoints | facebookresearch/sonata | CC BY-NC 4.0 |

The Sonata and Pointcept restrictions are inherited from the training data (HM3D, ArkitScenes)
rather than chosen by the model authors, so they cannot be waived upstream either.

## Pretrained weights

136 checkpoints are distributed through the model registry. Grouped by the release they were
converted from:

| source | author tag | license | count |
| --- | --- | --- | --- |
| guochengqian/PointNeXt, guochengqian/openpoints | `openpoints` | MIT | 36 |
| CGuangyan-BIT/PointGPT | `guangyan-chen` | MIT | 18 |
| antao97/dgcnn.pytorch | `an-tao` | MIT | 10 |
| lulutang0608/Point-BERT | `xumin-yu` | MIT | 8 |
| Pang-Yatian/Point-MAE | `yatian-pang` | MIT | 7 |
| Pointcept/Pointcept | `pointcept` | CC BY-NC 4.0 | 7 |
| open-mmlab/OpenPCDet | `openpcdet` | Apache-2.0 | 6 |
| ZrrSkywalker/Point-M2AE | `renrui-zhang` | MIT | 5 |
| LMD0311/PointMamba | `dingkang-liang` | Apache-2.0 | 5 |
| HuguesTHOMAS/KPConv-PyTorch | `hugues-thomas` | MIT | 4 |
| mit-han-lab/pvcnn, mit-han-lab/spvnas | `mit-han-lab` | MIT | 4 |
| ma-xu/pointMLP-pytorch | `xu-ma` | Apache-2.0 | 4 |
| Pointcept/Pointcept | `pointcept` | MIT | 4 |
| octree-nn/octformer | `octree-nn` | MIT | 3 |
| yanx27/Pointnet_Pointnet2_pytorch | `xu-yan` | MIT | 3 |
| facebookresearch/3detr | `fair` | Apache-2.0 | 3 |
| facebookresearch/sonata | `fair` | CC BY-NC 4.0 | 2 |
| facebookresearch/votenet | `fair` | MIT | 2 |
| filaPro/oneformer3d | `danila-rukhovich` | CC BY-NC 4.0 | 2 |
| DylanWusee/pointconv_pytorch | `wenxuan-wu` | MIT | 1 |
| QingyongHu/RandLA-Net | `tsung-han-wu` | MIT | 1 |
| happinesslz/LION | `zhe-liu` | Apache-2.0 | 1 |

Checkpoints are redistributed converted from the upstream release format to `safetensors`. Where a
release's class order differed from this project's label order, the head weights were permuted to
match. No checkpoint was retrained.

## Source code

Portions of this package were adapted from the implementations below. Each remains under its own
license; the adapted files carry a reference to their source.

| source | license | files |
| --- | --- | --- |
| open-mmlab/OpenPCDet | Apache-2.0 | `models/pointpillars.py`, `models/pointrcnn.py`, `models/second.py`, `models/voxelnext.py`, `models/voxel_mamba.py` |
| filaPro/oneformer3d | CC BY-NC 4.0 | `models/oneformer3d.py` |
| facebookresearch/3detr | Apache-2.0 | `models/detr3d.py` |
| facebookresearch/votenet | MIT | `models/votenet.py`, `lightning/callbacks.py` |
| Pointcept/Pointcept | MIT | `models/point_transformer_v3.py` |
| pyg-team/pytorch_geometric | MIT | `models/point_transformer.py` |
| Project-MONAI/MONAI | Apache-2.0 | `datasets/utils.py` |
| mit-han-lab/pvcnn | MIT | `utils/voxelization.py` |
| mit-han-lab/bevfusion | Apache-2.0 | `models/lion.py` |
| happinesslz/LION | Apache-2.0 | `models/lion.py` |
| gwenzhang/Voxel-Mamba | Apache-2.0 | `models/voxel_mamba.py`, `layers/vfe.py` |
| dvlab-research/VoxelNeXt | Apache-2.0 | `models/voxelnext.py` |
| sunjiahao1999/SPFormer | MIT | `models/spformer_unet.py` |
| lulutang0608/Point-BERT | MIT | `models/point_bert.py` |
| Pang-Yatian/Point-MAE | MIT | `models/point_mae.py` |
| ZrrSkywalker/Point-M2AE | MIT | `models/point_m2ae.py` |
| CGuangyan-BIT/PointGPT | MIT | `models/pointgpt.py` |
| LMD0311/PointMamba | Apache-2.0 | `models/point_mamba.py` |

Paths are relative to `src/torch_pointcloud/`.
