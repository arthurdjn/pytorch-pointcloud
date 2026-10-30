import pytest
import torch

from torch_pointcloud.layers import MaskedQueryDecoderLayer


def _layer(cross_attn_norm: bool = True) -> MaskedQueryDecoderLayer:
    torch.manual_seed(0)
    return MaskedQueryDecoderLayer(16, num_heads=2, mlp_dim=32, cross_attn_norm=cross_attn_norm).eval()


@pytest.mark.parametrize("cross_attn_norm", [True, False])
def test_masked_query_decoder_layer_scenes_of_different_sizes_are_independent(cross_attn_norm: bool) -> None:
    layer = _layer(cross_attn_norm)
    queries = torch.randn(2, 5, 16)
    sources = [torch.randn(7, 16), torch.randn(3, 16)]
    attn_masks = [torch.rand(5, 7) > 0.5, torch.rand(5, 3) > 0.5]
    with torch.no_grad():
        together = layer(queries, sources, attn_masks)
        alone = layer(queries[1:], sources[1:], attn_masks[1:])
    assert together.shape == (2, 5, 16)
    torch.testing.assert_close(together[1:], alone)


def test_masked_query_decoder_layer_ignores_blocked_sources() -> None:
    layer = _layer()
    queries = torch.randn(1, 4, 16)
    source = torch.randn(6, 16)
    attn_mask = torch.zeros(4, 6, dtype=torch.bool)
    attn_mask[:, 4:] = True
    changed = source.clone()
    changed[4:] = torch.randn(2, 16)
    with torch.no_grad():
        torch.testing.assert_close(layer(queries, [changed], [attn_mask]), layer(queries, [source], [attn_mask]))


def test_masked_query_decoder_layer_fully_blocked_row_attends_to_every_source() -> None:
    """A query whose mask blocks every source attends to all of them, rather than turning every query NaN."""
    layer = _layer()
    queries = torch.randn(1, 3, 16)
    source = torch.randn(5, 16)
    attn_mask = torch.rand(3, 5) > 0.5
    attn_mask[0] = True
    unblocked = attn_mask.clone()
    unblocked[0] = False
    with torch.no_grad():
        out = layer(queries, [source], [attn_mask])
        assert out.isfinite().all()
        torch.testing.assert_close(out, layer(queries, [source], [unblocked]))


def test_masked_query_decoder_layer_without_cross_attn_norm_has_no_extra_parameters() -> None:
    with_norm, without_norm = set(_layer(True).state_dict()), set(_layer(False).state_dict())
    assert with_norm - without_norm == {"norm0.weight", "norm0.bias"}
