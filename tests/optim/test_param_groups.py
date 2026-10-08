import fnmatch
from functools import partial

import pytest
import torch
from torch import nn

from torch_pointcloud.optim import generate_param_groups, param_groups


def test_filter_groups_by_glob() -> None:
    module = nn.ModuleDict({"block0": nn.Linear(4, 4), "head": nn.Linear(4, 2)})
    groups = generate_param_groups(
        module,
        layer_matches=[partial(fnmatch.fnmatchcase, pat="*block*")],
        match_types=["filter"],
        lr_values=[0.001],
    )
    assert len(groups) == 2  # block group + others
    assert groups[0]["lr"] == 0.001
    assert "lr" not in groups[1]
    assert {id(p) for p in groups[0]["params"]} == {id(p) for n, p in module.named_parameters() if "block" in n}
    assert {id(p) for p in groups[1]["params"]} == {id(p) for n, p in module.named_parameters() if "block" not in n}


def test_filter_supports_unix_wildcards() -> None:
    module = nn.ModuleDict({"block0": nn.Linear(2, 2), "blockhead": nn.Linear(2, 2)})
    # `block?.*` matches `block0.weight`/`block0.bias` but not `blockhead.*`.
    groups = generate_param_groups(
        module,
        layer_matches=[partial(fnmatch.fnmatchcase, pat="block?.*")],
        match_types=["filter"],
        lr_values=[0.1],
    )
    expected = {id(p) for n, p in module.named_parameters() if n.startswith("block0.")}
    assert {id(p) for p in groups[0]["params"]} == expected


def test_select_matcher_consumes_submodule_parameters() -> None:
    module = nn.ModuleDict({"encoder": nn.Linear(3, 3), "decoder": nn.Linear(3, 3)})
    groups = generate_param_groups(
        module,
        layer_matches=[lambda m: m["encoder"]],
        match_types=["select"],
        lr_values=[0.005],
    )
    assert len(groups) == 2
    assert groups[0]["lr"] == 0.005
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in module["encoder"].parameters()}
    assert {id(p) for p in groups[1]["params"]} == {id(p) for p in module["decoder"].parameters()}


def test_scalar_match_type_and_lr_broadcast() -> None:
    module = nn.ModuleDict({"block0": nn.Linear(2, 2), "block1": nn.Linear(2, 2), "head": nn.Linear(2, 2)})
    groups = generate_param_groups(
        module,
        layer_matches=[
            partial(fnmatch.fnmatchcase, pat="block0.*"),
            partial(fnmatch.fnmatchcase, pat="block1.*"),
        ],
        match_types="filter",
        lr_values=0.01,
    )
    assert len(groups) == 3  # 2 matched + others
    assert groups[0]["lr"] == 0.01 and groups[1]["lr"] == 0.01
    assert {id(p) for p in groups[2]["params"]} == {id(p) for p in module["head"].parameters()}


def test_partitions_all_params() -> None:
    module = nn.ModuleDict({"block0": nn.Linear(3, 3), "decoder": nn.Linear(3, 3)})
    groups = generate_param_groups(
        module,
        layer_matches=[partial(fnmatch.fnmatchcase, pat="*block*")],
        match_types="filter",
        lr_values=0.01,
    )
    grouped = sorted(id(p) for g in groups for p in g["params"])
    assert grouped == sorted(id(p) for p in module.parameters())


def test_include_others_false_drops_unmatched() -> None:
    module = nn.ModuleDict({"block0": nn.Linear(3, 3), "decoder": nn.Linear(3, 3)})
    groups = generate_param_groups(
        module,
        layer_matches=[partial(fnmatch.fnmatchcase, pat="*block*")],
        match_types="filter",
        lr_values=0.01,
        include_others=False,
    )
    assert len(groups) == 1
    assert {id(p) for p in groups[0]["params"]} == {id(p) for n, p in module.named_parameters() if "block" in n}


def test_invalid_match_type_raises() -> None:
    module = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="'select' or 'filter'"):
        generate_param_groups(
            module,
            layer_matches=[lambda m: m],
            match_types="weird",  # type: ignore[arg-type]
            lr_values=1.0,
        )


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(6, 8)
        self.block = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8)) for _ in range(2)])
        self.head = nn.Linear(8, 3)
        self.cls_token = nn.Parameter(torch.zeros(1, 8))
        self.frozen = nn.Linear(2, 2)
        self.frozen.requires_grad_(False)


def _names(model: nn.Module, group: dict) -> set:
    ids = {id(p) for p in group["params"]}
    return {n for n, p in model.named_parameters() if id(p) in ids}


def test_param_groups_default_is_a_single_decaying_group_of_trainable_params() -> None:
    model = _Model()
    groups = param_groups(model, weight_decay=0.05)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.05
    assert _names(model, groups[0]) == {n for n, p in model.named_parameters() if p.requires_grad}


def test_param_groups_no_decay_rule_excludes_biases_and_one_dim_params() -> None:
    model = _Model()
    decay, no_decay = param_groups(model, weight_decay=0.05, no_decay=True)
    assert no_decay["weight_decay"] == 0.0 and decay["weight_decay"] == 0.05
    assert _names(model, no_decay) == {
        "embedding.bias",
        "block.0.0.bias",
        "block.0.1.weight",
        "block.0.1.bias",
        "block.1.0.bias",
        "block.1.1.weight",
        "block.1.1.bias",
        "head.bias",
    }
    assert _names(model, decay) == {
        "embedding.weight",
        "block.0.0.weight",
        "block.1.0.weight",
        "head.weight",
        "cls_token",
    }


def test_param_groups_no_decay_keywords_and_callable() -> None:
    model = _Model()
    _, no_decay = param_groups(model, weight_decay=0.05, no_decay=["token"])
    assert "cls_token" in _names(model, no_decay)
    decay, no_decay = param_groups(model, weight_decay=0.05, no_decay=lambda name, param: name.startswith("head"))
    assert _names(model, no_decay) == {"head.weight", "head.bias"}


def test_param_groups_keyword_overrides_keep_pointcept_order_and_options() -> None:
    model = _Model()
    base, block = param_groups(model, weight_decay=0.05, overrides=[dict(keyword="block", lr=6e-4)])
    assert _names(model, block) == {n for n, _ in model.named_parameters() if "block" in n}
    assert block["lr"] == 6e-4 and block["weight_decay"] == 0.05 and "lr" not in base
    assert _names(model, base) == {"embedding.weight", "embedding.bias", "head.weight", "head.bias", "cls_token"}


def test_param_groups_first_matching_override_wins_and_match_callables_work() -> None:
    model = _Model()
    groups = param_groups(
        model, overrides=[dict(match=lambda n, p: n.endswith("weight"), lr=1.0), dict(keyword="block", lr=2.0)]
    )
    assert [g["lr"] for g in groups[1:]] == [1.0, 2.0]
    assert "block.0.0.weight" in _names(model, groups[1]) and "block.0.0.bias" in _names(model, groups[2])


def test_param_groups_splits_overrides_unless_they_fix_their_own_decay() -> None:
    model = _Model()
    groups = param_groups(model, weight_decay=0.05, no_decay=True, overrides=[dict(keyword="block", lr=6e-4)])
    assert [(g.get("lr"), g["weight_decay"]) for g in groups] == [(None, 0.05), (None, 0.0), (6e-4, 0.05), (6e-4, 0.0)]
    groups = param_groups(model, weight_decay=0.05, no_decay=True, overrides=[dict(keyword="block", weight_decay=0.0)])
    assert [(g.get("lr"), g["weight_decay"]) for g in groups] == [(None, 0.05), (None, 0.0), (None, 0.0)]


def test_param_groups_drops_empty_groups_and_validates_overrides() -> None:
    model = _Model()
    assert len(param_groups(model, overrides=[dict(keyword="nothing_matches")])) == 1
    with pytest.raises(ValueError, match="keyword"):
        param_groups(model, overrides=[dict(lr=1.0)])


def test_param_groups_feed_a_torch_optimizer() -> None:
    model = _Model()
    groups = param_groups(model, weight_decay=0.05, no_decay=True, overrides=[dict(keyword="block", lr=6e-4)])
    optimizer = torch.optim.AdamW(groups, lr=6e-3)
    assert [g["lr"] for g in optimizer.param_groups] == [6e-3, 6e-3, 6e-4, 6e-4]
    assert sum(len(g["params"]) for g in optimizer.param_groups) == sum(p.requires_grad for p in model.parameters())
