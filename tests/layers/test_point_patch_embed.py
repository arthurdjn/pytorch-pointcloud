import torch

from torch_pointcloud.layers.point_patch_embed import PointPatchEmbed


def test_point_patch_embed_without_bias_keeps_the_bias_of_the_last_layers() -> None:
    embed = PointPatchEmbed(embed_dim=32, local_channels=(16, 24), global_channels=(40,), bias=False)

    assert embed.local_mlp.lins[0].bias is None
    assert embed.local_mlp.lins[1].bias is not None
    assert embed.global_mlp.lins[0].bias is None
    assert embed.global_mlp.lins[1].bias is not None
    assert embed(torch.randn(2, 8, 4, 3)).shape == (2, 8, 32)
