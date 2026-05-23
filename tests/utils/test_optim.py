import fnmatch
from functools import partial

import pytest
from torch import nn

from torch_pointcloud.utils.optim import generate_param_groups


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
    """MONAI parity: scalar `match_types` / `lr_values` broadcast to all matchers."""
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
