from typing import List, Optional, Tuple, Union

import pytest
import torch
import torch.nn.functional as F

from torch_pointcloud.layers.conditional_heads import ConditionalLinear, CosineSimilarityHead


def test_cosine_similarity_head_forward_shape() -> None:
    head = CosineSimilarityHead(16, 8, num_classes=5, scale=10.0)
    logits = head(torch.randn(32, 16))
    assert logits.shape == (32, 5)


def test_cosine_similarity_head_scores_scaled_cosine_similarity() -> None:
    torch.manual_seed(0)
    head = CosineSimilarityHead(16, 8, num_classes=5, scale=10.0)
    x = torch.randn(32, 16)
    z = F.normalize(head.proj(x), dim=1)
    expected = 10.0 * (z @ head.class_embedding.T)
    assert torch.allclose(head(x), expected, atol=1e-5)
    assert head(x).abs().max() <= 10.0 + 1e-5


def test_cosine_similarity_head_class_embedding_is_a_unit_norm_buffer() -> None:
    head = CosineSimilarityHead(16, 8, num_classes=(5, 3), scale=1.0, conditions=["A", "B"])
    assert "class_embedding" in dict(head.named_buffers())
    assert "class_embedding" not in dict(head.named_parameters())
    assert set(head.state_dict()) == {"proj.weight", "proj.bias", "class_embedding"}
    assert head.class_embedding.shape == (8, 8)
    assert torch.allclose(head.class_embedding.norm(dim=1), torch.ones(8))


def test_cosine_similarity_head_condition_selects_its_rows() -> None:
    torch.manual_seed(0)
    head = CosineSimilarityHead(16, 8, num_classes=(5, 3), scale=2.0, conditions=["A", "B"])
    x = torch.randn(32, 16)
    z = F.normalize(head.proj(x), dim=1)
    logits_a = head(x, condition="A")
    logits_b = head(x, condition="B")
    assert logits_a.shape == (32, 5)
    assert logits_b.shape == (32, 3)
    assert torch.allclose(logits_a, 2.0 * (z @ head.class_embedding[:5].T), atol=1e-5)
    assert torch.allclose(logits_b, 2.0 * (z @ head.class_embedding[5:].T), atol=1e-5)


def test_cosine_similarity_head_ignores_condition_without_conditions() -> None:
    head = CosineSimilarityHead(16, 8, num_classes=5, scale=1.0)
    x = torch.randn(4, 16)
    assert torch.equal(head(x, condition="A"), head(x))


def test_cosine_similarity_head_trains_only_the_projection() -> None:
    head = CosineSimilarityHead(16, 8, num_classes=5, scale=1.0)
    head(torch.randn(4, 16)).sum().backward()
    assert {name for name, p in head.named_parameters() if p.grad is not None} == {"proj.weight", "proj.bias"}


def test_cosine_similarity_head_requires_a_known_condition() -> None:
    head = CosineSimilarityHead(16, 8, num_classes=(5, 3), scale=1.0, conditions=["A", "B"])
    x = torch.randn(4, 16)
    with pytest.raises(ValueError, match="requires a condition"):
        head(x)
    with pytest.raises(ValueError, match="Unknown condition 'C'"):
        head(x, condition="C")


@pytest.mark.parametrize(
    ("num_classes", "conditions"),
    [
        pytest.param(5, ["A", "B"], id="single-count-with-conditions"),
        pytest.param((5, 3), None, id="several-counts-without-conditions"),
        pytest.param((5, 3), ["A"], id="count-condition-mismatch"),
    ],
)
def test_cosine_similarity_head_validates_num_classes(
    num_classes: Union[int, Tuple[int, ...]],
    conditions: Optional[List[str]],
) -> None:
    with pytest.raises(ValueError):
        CosineSimilarityHead(16, 8, num_classes=num_classes, scale=1.0, conditions=conditions)


def test_cosine_similarity_head_from_embeddings_normalizes_one_table() -> None:
    class_embedding = torch.randn(5, 8)
    head = CosineSimilarityHead.from_embeddings(16, class_embedding, scale=10.0)
    assert head.num_classes == (5,)
    assert head.embed_dim == 8
    assert head.conditions is None
    assert torch.allclose(head.class_embedding, F.normalize(class_embedding, dim=1))


def test_cosine_similarity_head_from_embeddings_stacks_one_table_per_condition() -> None:
    tables = [torch.randn(5, 8), torch.randn(3, 8)]
    head = CosineSimilarityHead.from_embeddings(16, tables, scale=2.0, conditions=["A", "B"])
    x = torch.randn(32, 16)
    z = F.normalize(head.proj(x), dim=1)
    assert head.num_classes == (5, 3)
    assert torch.allclose(head(x, condition="B"), 2.0 * (z @ F.normalize(tables[1], dim=1).T), atol=1e-5)


def test_cosine_similarity_head_from_embeddings_keeps_the_tables_out_of_the_graph() -> None:
    class_embedding = torch.randn(5, 8, requires_grad=True)
    head = CosineSimilarityHead.from_embeddings(16, class_embedding, scale=1.0)
    assert head.class_embedding.grad_fn is None
    assert not head.class_embedding.requires_grad


def test_conditional_linear_condition_selects_its_layer() -> None:
    head = ConditionalLinear(16, num_classes=(5, 3), conditions=["A", "B"])
    x = torch.randn(32, 16)
    assert head(x, condition="A").shape == (32, 5)
    assert torch.equal(head(x, condition="B"), head[1](x))


def test_conditional_linear_keys_follow_the_conditions() -> None:
    head = ConditionalLinear(16, num_classes=(5, 3), conditions=["A", "B"])
    assert set(head.state_dict()) == {"0.weight", "0.bias", "1.weight", "1.bias"}
    assert head[1].weight.shape == (3, 16)


def test_conditional_linear_requires_a_known_condition() -> None:
    head = ConditionalLinear(16, num_classes=(5, 3), conditions=["A", "B"])
    x = torch.randn(4, 16)
    with pytest.raises(ValueError, match="requires a condition"):
        head(x)
    with pytest.raises(ValueError, match="Unknown condition 'C'"):
        head(x, condition="C")


def test_conditional_linear_validates_num_classes() -> None:
    with pytest.raises(ValueError, match="must name as many"):
        ConditionalLinear(16, num_classes=(5, 3), conditions=["A"])
