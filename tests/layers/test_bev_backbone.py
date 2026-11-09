import torch
import torch.nn.functional as F

from torch_pointcloud.layers.bev_backbone import BEVPyramidBackbone


def test_bev_pyramid_backbone_returns_one_map_per_level() -> None:
    backbone = BEVPyramidBackbone(4, [1, 1, 1], [2, 2, 2], [8, 16, 32], 12).eval()
    with torch.no_grad():
        levels = backbone(torch.randn(2, 4, 32, 32))
    assert backbone.num_bev_features == 12
    assert [tuple(level.shape) for level in levels] == [(2, 12, 16, 16), (2, 12, 8, 8), (2, 12, 4, 4)]


def test_bev_pyramid_backbone_adds_the_coarser_levels_top_down() -> None:
    torch.manual_seed(0)
    backbone = BEVPyramidBackbone(4, [1, 1], [2, 2], [8, 16], 12).eval()
    x = torch.randn(1, 4, 16, 16)
    with torch.no_grad():
        fine = backbone.blocks[0](x)
        coarse = backbone.blocks[1](fine)
        top = backbone.lateral_convs[1](coarse)
        merged = backbone.lateral_convs[0](fine) + F.interpolate(top, size=fine.shape[2:], mode="nearest")
        expected = [backbone.fpn_convs[0](merged), backbone.fpn_convs[1](top)]
        levels = backbone(x)
    assert all(torch.allclose(level, ref) for level, ref in zip(levels, expected))
