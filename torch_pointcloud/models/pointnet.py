import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers.activations import get_act_layer


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        bias: bool = False,
        act_layer: str = "relu",
    ) -> None:
        super(Block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act_layer = get_act_layer(act_layer)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


class PointNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        features_dim: int = 1024,
        dropout: float = 0.5,
        act_layer: str = "relu",
    ) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            Block(in_channels, 64, act_layer=act_layer),
            Block(64, 64, act_layer=act_layer),
            Block(64, 64, act_layer=act_layer),
            Block(64, 128, act_layer=act_layer),
            Block(128, features_dim, act_layer=act_layer),
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(features_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        return self.head(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)


# class ConVit(nn.Module):
#     def __init__(
#             self,
#             img_size=224,
#             patch_size=16,
#             in_chans=3,
#             num_classes=1000,
#             global_pool='token',
#             embed_dim=768,
#             depth=12,
#             num_heads=12,
#             mlp_ratio=4.,
#             qkv_bias=False,
#             drop_rate=0.,
#             pos_drop_rate=0.,
#             proj_drop_rate=0.,
#             attn_drop_rate=0.,
#             drop_path_rate=0.,
#             hybrid_backbone=None,
#             norm_layer=LayerNorm,
#             local_up_to_layer=3,
#             locality_strength=1.,
#             use_pos_embed=True,
#     ):
#         super().__init__()
#         assert global_pool in ('', 'avg', 'token')
#         embed_dim *= num_heads

# def _create_convit(variant, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     return build_model_with_cfg(ConVit, variant, pretrained, **kwargs)


# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'fixed_input_size': True,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#         **kwargs
#     }


# default_cfgs = generate_default_cfgs({
#     # ConViT
#     'convit_tiny.fb_in1k': _cfg(hf_hub_id='timm/'),
#     'convit_small.fb_in1k': _cfg(hf_hub_id='timm/'),
#     'convit_base.fb_in1k': _cfg(hf_hub_id='timm/')
# })


# @register_model
# def convit_tiny(pretrained=False, **kwargs) -> ConVit:
#     model_args = dict(
#         local_up_to_layer=10, locality_strength=1.0, embed_dim=48, num_heads=4)
#     model = _create_convit(variant='convit_tiny', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model
