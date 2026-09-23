# Third-Party Notices

`torch-pointcloud` is Apache-2.0 (see `LICENSE`). This file lists the third-party code, weights and
datasets used by the package and their licenses.

## Non-commercial material

Weights under a non-commercial license.

| material               | source                  | license      |
| ---------------------- | ----------------------- | ------------ |
| 5 Concerto checkpoints | Pointcept/Concerto      | CC BY-NC 4.0 |
| 2 Utonia checkpoints   | Pointcept/Utonia        | CC BY-NC 4.0 |
| 2 Sonata checkpoints   | facebookresearch/sonata | CC BY-NC 4.0 |

Sonata, Concerto and Utonia code is Apache-2.0; their weights are CC BY-NC 4.0.

## Pretrained weights

Checkpoints in the model registry, grouped by the release they were converted from. Each registry
entry records its license in the `license` field. The copyright column reproduces the notice of the upstream
`LICENSE` file, or of its source headers when the license text carries none; Apache-2.0 releases that publish
neither are marked as such.

| source                                          | author tag       | license      | count | copyright                                                                                                                  |
| ----------------------------------------------- | ---------------- | ------------ | ----- | -------------------------------------------------------------------------------------------------------------------------- |
| guochengqian/PointNeXt, guochengqian/openpoints | `openpoints`     | MIT          | 36    | Copyright (c) Guocheng Qian                                                                                                |
| CGuangyan-BIT/PointGPT                          | `guangyan-chen`  | MIT          | 18    | Copyright (c) 2022 PANG-Yatian, YUAN-Li                                                                                    |
| antao97/dgcnn.pytorch                           | `an-tao`         | MIT          | 10    | Copyright (c) 2020 An Tao                                                                                                  |
| Julie-tang00/Point-BERT                         | `xumin-yu`       | MIT          | 8     | Copyright (c) 2021 Xumin Yu                                                                                                |
| Pang-Yatian/Point-MAE                           | `yatian-pang`    | MIT          | 7     | Copyright (c) 2022 PANG-Yatian, YUAN-Li                                                                                    |
| open-mmlab/OpenPCDet                            | `openpcdet`      | Apache-2.0   | 6     | not stated upstream                                                                                                        |
| Pointcept/Concerto                              | `pointcept`      | CC BY-NC 4.0 | 5     | not stated upstream                                                                                                        |
| ZrrSkywalker/Point-M2AE                         | `renrui-zhang`   | MIT          | 5     | Copyright (c) 2022 Renrui Zhang                                                                                            |
| LMD0311/PointMamba                              | `dingkang-liang` | Apache-2.0   | 5     | not stated upstream                                                                                                        |
| HuguesTHOMAS/KPConv-PyTorch                     | `hugues-thomas`  | MIT          | 4     | Copyright (c) 2019 HuguesTHOMAS                                                                                            |
| mit-han-lab/pvcnn, mit-han-lab/spvnas           | `mit-han-lab`    | MIT          | 4     | Copyright (c) 2018 Zhijian Liu, Haotian Tang, Yujun Lin; Copyright (c) 2020 Zhijian Liu, Haotian Tang, Yujun Lin, Song Han |
| ma-xu/pointMLP-pytorch                          | `xu-ma`          | Apache-2.0   | 4     | not stated upstream                                                                                                        |
| Pointcept/Pointcept                             | `pointcept`      | MIT          | 4     | Copyright (c) 2023 Pointcept                                                                                               |
| octree-nn/octformer                             | `octree-nn`      | MIT          | 3     | Copyright (c) 2023 Peng-Shuai Wang                                                                                         |
| yanx27/Pointnet_Pointnet2_pytorch               | `xu-yan`         | MIT          | 3     | Copyright (c) 2019 benny                                                                                                   |
| facebookresearch/3detr                          | `fair`           | Apache-2.0   | 3     | Copyright (c) Facebook, Inc. and its affiliates                                                                            |
| Pointcept/Utonia                                | `pointcept`      | CC BY-NC 4.0 | 2     | not stated upstream                                                                                                        |
| facebookresearch/sonata                         | `fair`           | CC BY-NC 4.0 | 2     | Copyright (c) Meta Platforms, Inc. and affiliates                                                                          |
| facebookresearch/votenet                        | `fair`           | MIT          | 2     | Copyright (c) Facebook, Inc. and its affiliates                                                                            |
| DylanWusee/pointconv_pytorch                    | `wenxuan-wu`     | MIT          | 1     | Copyright (c) 2019 Wenxuan Wu                                                                                              |
| tsunghan-wu/RandLA-Net-pytorch                  | `tsung-han-wu`   | MIT          | 1     | Copyright (c) 2020 Tsunghan Wu                                                                                             |
| happinesslz/LION                                | `zhe-liu`        | Apache-2.0   | 1     | not stated upstream                                                                                                        |

## Training data

Licenses of the datasets the checkpoints were trained on.

| dataset                                         | terms                                                     | checkpoints |
| ----------------------------------------------- | --------------------------------------------------------- | ----------- |
| S3DIS (Stanford 2D-3D-S)                        | academic research agreement                               | 43          |
| ScanObjectNN                                    | terms of use, research only                               | 24          |
| ModelNet40                                      | academic research only                                    | 23          |
| ShapeNet, ShapeNetPart                          | non-commercial research and educational use               | 14          |
| ScanNet                                         | terms of use, non-commercial research and educational use | 13          |
| SemanticKITTI                                   | CC BY-NC-SA 4.0                                           | 4           |
| nuScenes                                        | CC BY-NC-SA 4.0                                           | 4           |
| KITTI                                           | CC BY-NC-SA 3.0                                           | 3           |
| SUN RGB-D                                       | research use                                              | 2           |
| HM3D, ArkitScenes (self-supervised pretraining) | non-commercial                                            | 6           |

## Source code

Files adapted from other implementations, with the copyright notice each source publishes. Paths are relative to
`src/torch_pointcloud/`.

| source                     | license    | files                                                                                                               | copyright                                               |
| -------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| open-mmlab/OpenPCDet       | Apache-2.0 | `models/pointpillars.py`, `models/pointrcnn.py`, `models/second.py`, `models/voxelnext.py`, `models/voxel_mamba.py` | not stated upstream                                     |
| facebookresearch/3detr     | Apache-2.0 | `models/detr3d.py`                                                                                                  | Copyright (c) Facebook, Inc. and its affiliates         |
| facebookresearch/votenet   | MIT        | `models/votenet.py`, `lightning/callbacks.py`                                                                       | Copyright (c) Facebook, Inc. and its affiliates         |
| Pointcept/Pointcept        | MIT        | `models/point_transformer_v3.py`                                                                                    | Copyright (c) 2023 Pointcept                            |
| pyg-team/pytorch_geometric | MIT        | `models/point_transformer.py`                                                                                       | Copyright (c) 2023 PyG Team <team@pyg.org>              |
| Project-MONAI/MONAI        | Apache-2.0 | `datasets/utils.py`                                                                                                 | Copyright (c) MONAI Consortium                          |
| mit-han-lab/pvcnn          | MIT        | `utils/voxelization.py`                                                                                             | Copyright (c) 2018 Zhijian Liu, Haotian Tang, Yujun Lin |
| mit-han-lab/bevfusion      | Apache-2.0 | `models/lion.py`                                                                                                    | Copyright 2018-2019 Open-MMLab. All rights reserved.    |
| happinesslz/LION           | Apache-2.0 | `models/lion.py`                                                                                                    | not stated upstream                                     |
| gwenzhang/Voxel-Mamba      | Apache-2.0 | `models/voxel_mamba.py`, `layers/vfe.py`                                                                            | not stated upstream                                     |
| JIA-Lab-research/VoxelNeXt | Apache-2.0 | `models/voxelnext.py`                                                                                               | not stated upstream                                     |
| sunjiahao1999/SPFormer     | MIT        | `models/spformer_unet.py`                                                                                           | Copyright (c) 2022 Jiahao Sun                           |
| Julie-tang00/Point-BERT    | MIT        | `models/point_bert.py`                                                                                              | Copyright (c) 2021 Xumin Yu                             |
| Pang-Yatian/Point-MAE      | MIT        | `models/point_mae.py`                                                                                               | Copyright (c) 2022 PANG-Yatian, YUAN-Li                 |
| ZrrSkywalker/Point-M2AE    | MIT        | `models/point_m2ae.py`                                                                                              | Copyright (c) 2022 Renrui Zhang                         |
| CGuangyan-BIT/PointGPT     | MIT        | `models/pointgpt.py`                                                                                                | Copyright (c) 2022 PANG-Yatian, YUAN-Li                 |
| LMD0311/PointMamba         | Apache-2.0 | `models/point_mamba.py`                                                                                             | not stated upstream                                     |

## Dependencies

Installed by `pip` next to the package, never bundled with it.

| dependency                                   | license                    |
| -------------------------------------------- | -------------------------- |
| `torch`                                      | BSD-3-Clause               |
| `torch-geometric`                            | MIT                        |
| `numpy`, `scipy`, `pandas`, `h5py`, `joblib` | BSD-3-Clause               |
| `safetensors`                                | Apache-2.0                 |
| `packaging`                                  | Apache-2.0 OR BSD-2-Clause |
| `pillow`                                     | MIT-CMU                    |
| `tqdm`                                       | MPL-2.0 AND MIT            |
| `typing-extensions`                          | PSF-2.0                    |
| `plyfile`                                    | GPL-3.0-or-later           |

## Optional dependencies

Installed separately, only for the architectures that need them.

| dependency                                                                                       | license                     |
| ------------------------------------------------------------------------------------------------ | --------------------------- |
| `pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster`                                      | MIT                         |
| `spconv`                                                                                         | Apache-2.0                  |
| `torchsparse`                                                                                    | MIT                         |
| `ocnn`, `dwconv`                                                                                 | MIT                         |
| `flash-attn`                                                                                     | BSD-3-Clause                |
| `mamba-ssm`, `causal-conv1d`                                                                     | Apache-2.0 AND BSD-3-Clause |
| `sptr` (JIA-Lab-research/SparseTransformer, installed from the arthurdjn/SparseTransformer fork) | Apache-2.0                  |
| `lightning`, `torchmetrics`                                                                      | Apache-2.0                  |
