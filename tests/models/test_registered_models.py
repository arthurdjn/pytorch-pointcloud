import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
import torch
import torch.nn as nn

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models._registry import _REGISTERED_MODELS
from torch_pointcloud.utils.imports import (
    _DWCONV_AVAILABLE,
    _FLASH_ATTN_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _OCNN_AVAILABLE,
    _SPCONV_AVAILABLE,
    _SPTR_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
    _TORCHSPARSE_AVAILABLE,
)
from torch_pointcloud.utils.octree import build_octree

SNAPSHOTS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"

CLASSIFICATION_MODELS = [
    "dgcnn.modelnet40-1024.an-tao",
    "dgcnn.modelnet40-2048.an-tao",
    "kpfcnn.modelnet40",
    "octformer-base.modelnet40.octree-nn",
    "pointcnn-base",
    "pointconv-density-base",
    "pointconv-density-base.modelnet40.wenxuan-wu",
    "point-bert-base.modelnet40.xumin-yu",
    "point-bert-base.modelnet40-4k.xumin-yu",
    "point-bert-base.modelnet40-8k.xumin-yu",
    "point-bert-base.scanobjectnn-hardest.xumin-yu",
    "point-bert-base.scanobjectnn-objbg.xumin-yu",
    "point-bert-base.scanobjectnn-objonly.xumin-yu",
    "point-m2ae-base.modelnet40.renrui-zhang",
    "point-m2ae-base.scanobjectnn-hardest.renrui-zhang",
    "point-m2ae-base.scanobjectnn-objbg.renrui-zhang",
    "point-mae-base.modelnet40.yatian-pang",
    "point-mae-base.modelnet40-8k.yatian-pang",
    "point-mae-base.scanobjectnn-hardest.yatian-pang",
    "point-mae-base.scanobjectnn-objbg.yatian-pang",
    "point-mae-base.scanobjectnn-objonly.yatian-pang",
    "point-mamba-base.modelnet40.dingkang-liang",
    "point-mamba-base.scanobjectnn.dingkang-liang",
    "point-mamba-base.scanobjectnn-nobg.dingkang-liang",
    "point-mamba-base.scanobjectnn-augmentedrot-scale75.dingkang-liang",
    "point-transformer.modelnet40",
    "pointgpt-s.modelnet40.guangyan-chen",
    "pointgpt-s.modelnet40-8k.guangyan-chen",
    "pointgpt-s.scanobjectnn-hardest.guangyan-chen",
    "pointgpt-s.scanobjectnn-objbg.guangyan-chen",
    "pointgpt-s.scanobjectnn-objonly.guangyan-chen",
    "pointgpt-b.modelnet40.guangyan-chen",
    "pointgpt-b.modelnet40-8k.guangyan-chen",
    "pointgpt-b.scanobjectnn-hardest.guangyan-chen",
    "pointgpt-b.scanobjectnn-objbg.guangyan-chen",
    "pointgpt-b.scanobjectnn-objonly.guangyan-chen",
    "pointgpt-l.modelnet40.guangyan-chen",
    "pointgpt-l.modelnet40-8k.guangyan-chen",
    "pointgpt-l.scanobjectnn-hardest.guangyan-chen",
    "pointgpt-l.scanobjectnn-objbg.guangyan-chen",
    "pointgpt-l.scanobjectnn-objonly.guangyan-chen",
    "pointmlp-base",
    "pointmlp-base.modelnet40.xu-ma",
    "pointmlp-base.scanobjectnn.xu-ma",
    "pointmlp-elite",
    "pointmlp-elite.modelnet40.xu-ma",
    "pointmlp-elite.scanobjectnn.xu-ma",
    "pointnet.modelnet40",
    "pointnet2.modelnet40.openpoints",
    "pointnet2.scanobjectnn.openpoints",
    "pointnet2-msg.modelnet40.xu-yan",
    "pointnet2-ssg.modelnet40.xu-yan",
    "pointnext-base",
    "pointnext-lg",
    "pointnext-sm",
    "pointnext-sm-c64.modelnet40.openpoints",
    "pointnext-sm.scanobjectnn.openpoints",
    "pointnext-xl",
]
BASE_MODELS = [
    "concerto-base.pretrain.pointcept",
    "concerto-large.pretrain.pointcept",
    "concerto-small.pretrain.pointcept",
    "concerto-tiny.pretrain.pointcept",
    "point-bert-base.dvae.xumin-yu",
    "point-bert-base.pretrain.xumin-yu",
    "point-m2ae-base.pretrain.renrui-zhang",
    "point-mae-base.pretrain.yatian-pang",
    "point-mamba-base.pretrain.dingkang-liang",
    "pointgpt-s.pretrain.guangyan-chen",
    "pointgpt-b.pretrain.guangyan-chen",
    "pointgpt-l.pretrain.guangyan-chen",
    "sonata-base.pretrain.fair",
    "spformer-unet.scannet",
    "utonia.pretrain.pointcept",
]
SEGMENTATION_MODELS = [
    "concerto-large-lp.scannet20.pointcept",
    "utonia-lp.scannet20.pointcept",
    "dgcnn.shapenetpart.an-tao",
    "dgcnn.s3dis-area1.an-tao",
    "dgcnn.s3dis-area2.an-tao",
    "dgcnn.s3dis-area3.an-tao",
    "dgcnn.s3dis-area4.an-tao",
    "dgcnn.s3dis-area5.an-tao",
    "dgcnn.s3dis-area6.an-tao",
    "dgcnn.scannet20.an-tao",
    "kpfcnn-base.s3dis.hugues-thomas",
    "kpfcnn-base-sm.s3dis.hugues-thomas",
    "kpfcnn-base-deform.s3dis.hugues-thomas",
    "kpfcnn-base-sm-deform.s3dis.hugues-thomas",
    "octformer-lg",
    "octformer-sm",
    "octformer-base.scannet20.octree-nn",
    "octformer-base.scannet200.octree-nn",
    "oneformer3d-base.s3dis-area5.danila-rukhovich",
    "oneformer3d-base.scannet20.danila-rukhovich",
    "oneformer3d-base.scannet200.danila-rukhovich",
    "point-mae-base.shapenetpart.yatian-pang",
    "point-m2ae-base.shapenetpart.renrui-zhang",
    "point-transformer.s3dis-area5",
    "point-transformer.scannet20",
    "pointcnn-base",
    "pointmlp-base",
    "pointmlp-elite",
    "pointnet.s3dis-area5",
    "pointnet.shapenetpart",
    "pointnet2.s3dis-area1.openpoints",
    "pointnet2.s3dis-area2.openpoints",
    "pointnet2.s3dis-area3.openpoints",
    "pointnet2.s3dis-area4.openpoints",
    "pointnet2.s3dis-area5.openpoints",
    "pointnet2.s3dis-area6.openpoints",
    "pointnet2.s3dis-area5.xu-yan",
    "ptv2-base.scannet20",
    "ptv2-base.scannet200",
    "ptv3-base.scannet20.pointcept",
    "ptv3-base.scannet200.pointcept",
    "ptv3-base.s3dis-area5.pointcept",
    "pvcnn.s3dis-area5.mit-han-lab",
    "pvcnn2.s3dis-area5",
    "pointnext-base",
    "pointnext-base.s3dis-area1.openpoints",
    "pointnext-base.s3dis-area2.openpoints",
    "pointnext-base.s3dis-area3.openpoints",
    "pointnext-base.s3dis-area4.openpoints",
    "pointnext-base.s3dis-area5.openpoints",
    "pointnext-base.s3dis-area6.openpoints",
    "pointnext-lg",
    "pointnext-lg.s3dis-area1.openpoints",
    "pointnext-lg.s3dis-area2.openpoints",
    "pointnext-lg.s3dis-area3.openpoints",
    "pointnext-lg.s3dis-area4.openpoints",
    "pointnext-lg.s3dis-area5.openpoints",
    "pointnext-lg.s3dis-area6.openpoints",
    "pointnext-sm",
    "pointnext-sm.s3dis-area1.openpoints",
    "pointnext-sm.s3dis-area2.openpoints",
    "pointnext-sm.s3dis-area3.openpoints",
    "pointnext-sm.s3dis-area4.openpoints",
    "pointnext-sm.s3dis-area5.openpoints",
    "pointnext-sm.s3dis-area6.openpoints",
    "pointnext-sm.shapenetpart.openpoints",
    "pointnext-sm-c64.shapenetpart.openpoints",
    "pointnext-sm-c160.shapenetpart.openpoints",
    "pointnext-xl",
    "pointnext-xl.s3dis-area1.openpoints",
    "pointnext-xl.s3dis-area2.openpoints",
    "pointnext-xl.s3dis-area3.openpoints",
    "pointnext-xl.s3dis-area4.openpoints",
    "pointnext-xl.s3dis-area5.openpoints",
    "pointnext-xl.s3dis-area6.openpoints",
    "randlanet.semantickitti.tsung-han-wu",
    "sonata-lp.scannet20.fair",
    "spformer-unet.scannet20",
    "sphereformer.nuscenes",
    "sphereformer.semantickitti",
    "spunet-v1m1.scannet20.pointcept",
    "spvcnn-30gmacs.semantickitti.mit-han-lab",
    "spvcnn-47gmacs.semantickitti.mit-han-lab",
    "spvcnn-119gmacs.semantickitti.mit-han-lab",
]
DETECTION_MODELS = [
    "3detr-m.scannet.fair",
    "3detr.scannet.fair",
    "3detr.sunrgbd.fair",
    "lion-mamba.nuscenes.zhe-liu",
    "pointpillars-multihead.nuscenes.openpcdet",
    "pointpillars.kitti.openpcdet",
    "pointrcnn.kitti.openpcdet",
    "second-multihead.nuscenes.openpcdet",
    "second.kitti.openpcdet",
    "votenet.scannet.fair",
    "votenet.sunrgbd.fair",
    "voxel-mamba.waymo",
    "voxelnext.nuscenes.openpcdet",
]


def _skip_if_model_deps_missing(model_name: str) -> None:
    if model_name.startswith("point-mamba") and not _MAMBA_SSM_AVAILABLE:
        pytest.skip("mamba_ssm is not installed")
    if model_name.startswith("octformer") and not _DWCONV_AVAILABLE:
        pytest.skip("dwconv is not installed")
    if model_name.startswith(("sonata", "concerto", "utonia", "ptv3")) and not _FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn is not installed")
    if model_name.startswith("spvcnn") and not _TORCHSPARSE_AVAILABLE:
        pytest.skip("torchsparse is not installed")
    if model_name.startswith(("spunet", "oneformer3d", "spformer")) and not _SPCONV_AVAILABLE:
        pytest.skip("spconv is not installed")
    if model_name.startswith("oneformer3d") and not _TORCH_SCATTER_AVAILABLE:
        pytest.skip("torch_scatter is not installed")
    if model_name.startswith(("voxelnext", "voxel-mamba")) and not _SPCONV_AVAILABLE:
        pytest.skip("spconv is not installed")
    if model_name.startswith("voxel-mamba") and not _MAMBA_SSM_AVAILABLE:
        pytest.skip("mamba_ssm is not installed")
    if model_name.startswith(("3detr", "pointrcnn")) and not _TORCH_CLUSTER_AVAILABLE:
        pytest.skip("torch_cluster is not installed")
    if model_name.startswith("lion") and not (_MAMBA_SSM_AVAILABLE and _SPCONV_AVAILABLE):
        pytest.skip("mamba_ssm or spconv is not installed")
    if model_name.startswith("sphereformer") and not (_SPCONV_AVAILABLE and _SPTR_AVAILABLE):
        pytest.skip("spconv or sptr is not installed")


# Models whose `forward` cannot run on the synthetic `data_factory` input — they expect
# serialized / octree-encoded inputs that the factory doesn't emit yet.
_UNSUPPORTED_BY_DATA_FACTORY = frozenset(
    {
        "octformer-base.modelnet40.octree-nn",
        "octformer-lg",
        "sonata-lp.scannet20.fair",
        "concerto-large-lp.scannet20.pointcept",
        "utonia-lp.scannet20.pointcept",
    }
)


def _skip_if_model_not_runnable(model_name: str) -> None:
    """Skip everything that can't be exercised in `test_model_forward` on this box."""
    _skip_if_model_deps_missing(model_name)
    if (
        model_name.startswith(("point-mamba", "spvcnn", "spformer", "spunet", "oneformer3d", "octformer", "ptv3"))
        and not torch.cuda.is_available()
    ):
        pytest.skip(f"{model_name} requires CUDA, none available")
    if model_name.startswith("sphereformer"):
        if not torch.cuda.is_available():
            pytest.skip(f"{model_name} requires CUDA, none available")
        pytest.importorskip("sptr")
    if model_name.startswith("ptv3") and not _FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn is not installed")
    if model_name in _UNSUPPORTED_BY_DATA_FACTORY:
        pytest.skip("Synthetic data factory doesn't support this model's input format yet")
    if model_name.startswith("randlanet."):
        # /4..16x decimation drops below K=16 on the synthetic 512+768 cloud; real clouds
        # are ~10^5+ points so this is purely a test-data artifact.
        pytest.skip("Synthetic test cloud is too small for /4 decimation with K=16.")


def _check_output_tensor(output: object, *, label: str, expected_shape: tuple[int, int], allow_nonfinite: bool) -> None:
    """Check a per-point/per-object logits tensor: float dtype, finite, expected shape."""
    assert isinstance(output, torch.Tensor), f"{label}: expected Tensor, got {type(output).__name__}"
    assert torch.is_floating_point(output), f"{label}: non-float dtype {output.dtype}"
    if not allow_nonfinite:
        assert torch.isfinite(output).all().item(), f"{label}: output contains NaN or Inf"
    assert tuple(output.shape) == expected_shape, (
        f"{label}: output shape {tuple(output.shape)} != expected {expected_shape}"
    )


def _check_output_dict(output: dict, *, label: str) -> None:
    """Check a OneFormer3D-style output: a dict of per-scene lists, not per-point logits.

    `cls_preds` is over instance classes (independent of the semantic `num_classes`), so
    we only check structural validity, not a specific class count.
    """
    cls_preds = output["cls_preds"]
    assert isinstance(cls_preds, list) and len(cls_preds) >= 1, f"{label}: empty cls_preds"
    last_dims = {cls_pred.shape[-1] for cls_pred in cls_preds}
    assert len(last_dims) == 1, f"{label}: inconsistent cls_preds class dims {last_dims}"
    for cls_pred in cls_preds:
        assert cls_pred.ndim == 2, f"{label}: cls_preds should be 2D, got {cls_pred.ndim}D"
        assert torch.is_floating_point(cls_pred), f"{label}: non-float cls_preds {cls_pred.dtype}"
        assert torch.isfinite(cls_pred).all().item(), f"{label}: cls_preds has NaN or Inf"


def _check_forward_output(
    output: object,
    *,
    model_name: str,
    task: str,
    expected_shape: tuple[int, int],
) -> None:
    """Assert that `output` is structurally valid.

    We deliberately don't snapshot values: random-weight numerics are fragile across
    torch / CUDA / scatter-op versions and would flap without catching real model bugs.
    """
    label = f"{model_name} ({task})"
    if isinstance(output, dict):
        _check_output_dict(output, label=label)
        return
    # KDE-based density estimation in `pointconv-density` is numerically unstable with
    # random weights at small `in_channels`. Pretrained weights converge to finite output.
    _check_output_tensor(
        output,
        label=label,
        expected_shape=expected_shape,
        allow_nonfinite=model_name.startswith("pointconv-density"),
    )


def _check_architecture_or_regen(
    model: nn.Module,
    model_name: str,
    task: str,
    models_dir: Path,
    force_regen: bool,
) -> None:
    # Skip uninitialized lazy parameters since their shape is only known after a forward pass
    parameters = [p for p in model.parameters() if not isinstance(p, nn.parameter.UninitializedParameter)]
    state_dict = {
        k: list(v.shape)
        for k, v in model.state_dict().items()
        if not isinstance(v, nn.parameter.UninitializedParameter)
    }
    metadata = {
        "num_params": sum(p.numel() for p in parameters),
        "state_dict": state_dict,
    }

    expected_path = models_dir / f"{model_name}_{task}.json"
    if force_regen or not expected_path.exists():
        snapshot_path = SNAPSHOTS_DIR / f"{model_name}_{task}.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(metadata, indent=2) + "\n")
        pytest.fail(f"Regenerated {snapshot_path.as_posix()!r}")

    expected = json.loads(expected_path.read_text())

    assert metadata["num_params"] == expected["num_params"], (
        f"Parameter count changed for {model_name!r} ({task}): "
        f"expected {expected['num_params']}, got {metadata['num_params']}"
    )
    assert metadata["state_dict"] == expected["state_dict"], f"State-dict structure changed for {model_name!r} ({task})"


@pytest.mark.parametrize(
    "task,expected_models",
    [
        ("base", BASE_MODELS),
        ("classification", CLASSIFICATION_MODELS),
        ("segmentation", SEGMENTATION_MODELS),
        ("detection", DETECTION_MODELS),
    ],
)
def test_list_models(task: str, expected_models: List[str]) -> None:
    """Test that the list of models is correct."""
    models = list_models(task=task)  # type: ignore[arg-type]

    if set(models) != set(expected_models):
        missing_models = set(models) - set(expected_models)
        extra_models = set(expected_models) - set(models)
        err_msg = f"Expected {len(expected_models)} {task} models, got {len(models)}. "
        if missing_models:
            err_msg += f"\nMissing models: {', '.join(sorted(missing_models))}"
        if extra_models:
            err_msg += f"\nExtra models: {', '.join(sorted(extra_models))}"

        raise AssertionError(err_msg)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification_architecture(model_name: str, force_regen: bool, models_dir_factory: Any) -> None:
    """Test that the architecture of all registered segmentation models is correct.
    This test will only verify that the state-dict structure of the model matches the expected structure,
    but will not verify that the content of the weights are correct.

    This test is useful to catch accidental architecture changes in the models (e.g. renaming a parameter or module),
    and is faster than a full forward pass + weight loading.

    To regenerate the expected architecture as JSON files, run

    ```bash
    uv run pytest tests/models/test_registered_models.py -k test_classification_architecture --force-regen
    ```
    """
    models_dir = models_dir_factory("*.json")

    _skip_if_model_deps_missing(model_name)
    model = create_model(model_name, task="classification", in_channels=3, num_classes=10)
    _check_architecture_or_regen(
        model,
        model_name,
        task="classification",
        models_dir=models_dir,
        force_regen=force_regen,
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", SEGMENTATION_MODELS)
def test_segmentation_architecture(model_name: str, force_regen: bool, models_dir_factory: Any) -> None:
    """Test that the architecture of all registered segmentation models is correct.
    This test will only verify that the state-dict structure of the model matches the expected structure,
    but will not verify that the content of the weights are correct.

    This test is useful to catch accidental architecture changes in the models (e.g. renaming a parameter or module),
    and is faster than a full forward pass + weight loading.

    To regenerate the expected architecture as JSON files, run

    ```bash
    uv run --no-sync pytest tests/models/test_registered_models.py -k test_segmentation_architecture --force-regen
    ```
    """
    _skip_if_model_deps_missing(model_name)
    models_dir = models_dir_factory("*.json")

    model = create_model(model_name, task="segmentation", in_channels=3, num_classes=10)
    _check_architecture_or_regen(
        model,
        model_name,
        task="segmentation",
        models_dir=models_dir,
        force_regen=force_regen,
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", DETECTION_MODELS)
def test_detection_architecture(model_name: str, force_regen: bool, models_dir_factory: Any) -> None:
    """Test that the architecture of all registered detection models is correct.
    This test will only verify that the state-dict structure of the model matches the expected structure,
    but will not verify that the content of the weights are correct.

    This test is useful to catch accidental architecture changes in the models (e.g. renaming a parameter or module),
    and is faster than a full forward pass + weight loading.

    To regenerate the expected architecture as JSON files, run

    ```bash
    uv run --no-sync pytest tests/models/test_registered_models.py -k test_detection_architecture --force-regen
    ```
    """
    _skip_if_model_deps_missing(model_name)
    models_dir = models_dir_factory("*.json")

    model = create_model(model_name, task="detection")
    assert isinstance(model.num_features, int)
    _check_architecture_or_regen(
        model,
        model_name,
        task="detection",
        models_dir=models_dir,
        force_regen=force_regen,
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", BASE_MODELS)
def test_base_architecture(model_name: str, force_regen: bool, models_dir_factory: Any) -> None:
    """Test that the architecture of all registered base models is correct.

    Base models (SSL encoders and pretraining heads) carry no task wrapper and, when they ship without
    pretrained weights, have no pretrained-weight regression test, so this random-weight state-dict snapshot
    is their only guard against accidental architecture changes (renamed parameters, changed shapes) caused
    by refactors to shared layers, transforms or utilities.

    To regenerate the expected architecture as JSON files, run

    ```bash
    uv run --no-sync pytest tests/models/test_registered_models.py -k test_base_architecture --force-regen
    ```
    """
    _skip_if_model_deps_missing(model_name)
    models_dir = models_dir_factory("*.json")

    model = create_model(model_name, task="base")
    _check_architecture_or_regen(
        model,
        model_name,
        task="base",
        models_dir=models_dir,
        force_regen=force_regen,
    )


def _make_voxel_inputs(lengths: torch.Tensor, grid_size: int = 16) -> Dict[str, torch.Tensor]:
    """Synthetic sparse-voxel inputs for spconv / superpoint models.

    `pos_grid` holds unique integer voxel coords per scene (no duplicates, so submanifold
    convs are well defined); each point is its own voxel (`inverse`) and superpoint, re-based
    per scene so ids are globally disjoint. Point-based models ignore these keys.
    """
    gen = torch.Generator().manual_seed(42)
    grid_parts: List[torch.Tensor] = []
    superpoint_parts: List[torch.Tensor] = []
    running = 0
    for length in lengths.tolist():
        flat = torch.randperm(grid_size**3, generator=gen)[:length]
        grid_parts.append(torch.stack([flat // grid_size**2, (flat // grid_size) % grid_size, flat % grid_size], 1))
        superpoint_parts.append(torch.arange(length) + running)
        running += length
    return {
        "pos_grid": torch.cat(grid_parts).long(),
        "superpoint": torch.cat(superpoint_parts).long(),
        "inverse": torch.arange(int(lengths.sum())),
    }


@pytest.fixture
def data_factory() -> Callable[[int, int], Dict[str, Any]]:
    def create_data(
        in_channels: int,
        spatial_dim: int = 3,
        num_categories: int = 0,
        lengths: tuple = (512, 768),
    ) -> Dict[str, Any]:
        torch.manual_seed(42)
        lengths_tensor = torch.tensor(list(lengths))
        pos = torch.randn(int(lengths_tensor.sum()), spatial_dim)
        x = torch.randn(int(lengths_tensor.sum()), in_channels) if in_channels > 0 else None
        batch = torch.repeat_interleave(torch.arange(len(lengths_tensor)), lengths_tensor)
        cls_onehot = torch.zeros(len(lengths_tensor), num_categories) if num_categories > 0 else None
        voxel = _make_voxel_inputs(lengths_tensor)

        octree = None
        if _OCNN_AVAILABLE:
            octree = build_octree(
                pos=pos,
                batch=batch,
                x=x,
                batch_size=len(lengths),
            )
            octree.construct_all_neigh()
            x = octree.features[octree.depth]
            pos = octree.points[octree.depth]

        return dict(
            x=x,
            pos=pos,
            octree=octree,
            depth=octree.depth if octree is not None else None,
            batch=batch,
            cls_onehot=cls_onehot,
            category=cls_onehot,
            **voxel,
        )

    return create_data


# The Point-MAE / Point-M2AE part-segmentation heads require a uniform number of points per sample
# (ragged packed batches raise ValueError), so they get a uniform synthetic cloud.
UNIFORM_POINT_MODELS = ("point-mae-base.shapenetpart.yatian-pang", "point-m2ae-base.shapenetpart.renrui-zhang")


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize(
    "model_name,task",
    [
        *[(model, "classification") for model in CLASSIFICATION_MODELS],
        *[(model, "segmentation") for model in SEGMENTATION_MODELS],
    ],
)
def test_model_forward(model_name: str, task: str, data_factory: Callable) -> None:
    _skip_if_model_not_runnable(model_name)

    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(model_name, task=task, in_channels=3, num_classes=num_classes)  # type: ignore[call-overload]
    data = data_factory(
        in_channels=model.in_channels,
        spatial_dim=getattr(model, "spatial_dim", 3),
        num_categories=getattr(model, "num_categories", 0),
        lengths=(512, 512) if model_name in UNIFORM_POINT_MODELS else (512, 768),
    )
    expected_rows = int(data["batch"].max().item()) + 1 if task == "classification" else data["pos"].shape[0]

    sig = inspect.signature(model.forward)
    kwargs = {a: data[a] for a in sig.parameters if a != "self" and a in data}
    kwargs = {k: v.to(device) if hasattr(v, "to") else v for k, v in kwargs.items()}
    model = model.to(device)

    output = model(**kwargs)
    _check_forward_output(output, model_name=model_name, task=task, expected_shape=(expected_rows, num_classes))

    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize(
    "model_name,task",
    [
        *[(model, "classification") for model in CLASSIFICATION_MODELS],
        *[(model, "segmentation") for model in SEGMENTATION_MODELS],
    ],
)
def test_model_headless_forward_returns_features(model_name: str, task: str, data_factory: Callable) -> None:
    """`num_classes=0` drops the head: `forward` returns `(rows, num_features)` features."""
    _skip_if_model_not_runnable(model_name)
    if model_name.startswith("oneformer3d"):
        pytest.skip("OneFormer3D has no headless mode (the query decoder is the model)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(model_name, task=task, in_channels=3, num_classes=0)  # type: ignore[call-overload]
    assert isinstance(model.head, nn.Identity), f"{model_name}: head is {type(model.head).__name__}, not Identity"
    assert isinstance(model.num_features, int), f"{model_name}: num_features is {type(model.num_features).__name__}"
    data = data_factory(
        in_channels=model.in_channels,
        spatial_dim=getattr(model, "spatial_dim", 3),
        num_categories=getattr(model, "num_categories", 0),
        lengths=(512, 512) if model_name in UNIFORM_POINT_MODELS else (512, 768),
    )
    expected_rows = int(data["batch"].max().item()) + 1 if task == "classification" else data["pos"].shape[0]

    sig = inspect.signature(model.forward)
    kwargs = {a: data[a] for a in sig.parameters if a != "self" and a in data}
    kwargs = {k: v.to(device) if hasattr(v, "to") else v for k, v in kwargs.items()}
    model = model.to(device)

    output = model(**kwargs)
    _check_forward_output(output, model_name=model_name, task=task, expected_shape=(expected_rows, model.num_features))

    if device == "cuda":
        torch.cuda.empty_cache()


def _make_detection_inputs(model_name: str, in_channels: int, n_per_scene: int = 256) -> Dict[str, Any]:
    """Two tiny packed scenes: indoor-scale positions by default, KITTI-range positions for PointRCNN
    (whose `in_channels` counts the three coordinate channels, so features carry `in_channels - 3`)."""
    torch.manual_seed(42)
    n = 2 * n_per_scene
    batch = torch.arange(2).repeat_interleave(n_per_scene)
    if model_name.startswith("pointrcnn"):
        low = torch.tensor([0.0, -40.0, -3.0])
        high = torch.tensor([70.4, 40.0, 1.0])
        pos = torch.rand(n, 3) * (high - low) + low
        x = torch.rand(n, in_channels - 3)
        return {"x": x, "pos": pos, "batch": batch}
    pos = torch.rand(n, 3) * 4.0
    return {"x": torch.rand(n, in_channels) if in_channels > 0 else None, "pos": pos, "batch": batch}


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize(
    "model_name,model_kwargs",
    [
        pytest.param(
            "votenet.scannet.fair", dict(sa_npoints=[128, 64, 32, 16], num_proposal=32), id="votenet.scannet.fair"
        ),
        pytest.param(
            "votenet.sunrgbd.fair", dict(sa_npoints=[128, 64, 32, 16], num_proposal=32), id="votenet.sunrgbd.fair"
        ),
        pytest.param("3detr.scannet.fair", dict(preenc_npoints=128, num_queries=32), id="3detr.scannet.fair"),
        pytest.param("3detr-m.scannet.fair", dict(preenc_npoints=128, num_queries=32), id="3detr-m.scannet.fair"),
        pytest.param("3detr.sunrgbd.fair", dict(preenc_npoints=128, num_queries=32), id="3detr.sunrgbd.fair"),
        pytest.param(
            "pointrcnn.kitti.openpcdet",
            dict(sa_npoints=[128, 64, 32, 16], num_sampled_points=64, nms_post_maxsize=16),
            id="pointrcnn.kitti.openpcdet",
        ),
    ],
)
def test_detection_model_forward(model_name: str, model_kwargs: Dict[str, Any]) -> None:
    """CPU forward + decode smoke for the point-based detectors.

    `model_kwargs` only shrinks sampling sizes (`sa_npoints`, `num_proposal`, `preenc_npoints`, ...), which
    keeps the state-dict structure identical to the registered configuration while the forward stays fast
    on a few hundred CPU points. The grid-based detectors (pointpillars, second, voxelnext, voxel-mamba,
    lion) take voxelized inputs (and most need spconv / mamba_ssm); their per-model test files cover them.
    """
    _skip_if_model_deps_missing(model_name)
    model = create_model(model_name, task="detection", **model_kwargs).eval()
    data = _make_detection_inputs(model_name, in_channels=model.in_channels)

    with torch.no_grad():
        output = model(data["x"], data["pos"], data["batch"])
        detections = model.decode(output)

    assert isinstance(output, dict) and output, f"{model_name}: expected a non-empty output dict"
    for key, value in output.items():
        assert isinstance(value, torch.Tensor), f"{model_name}: output[{key!r}] is not a Tensor"
        assert torch.isfinite(value).all().item(), f"{model_name}: output[{key!r}] contains NaN or Inf"

    num_boxes = detections["boxes"].shape[0]
    assert detections["boxes"].shape == (num_boxes, 7)
    assert detections["scores"].shape == (num_boxes,)
    assert detections["labels"].shape == (num_boxes,)
    assert detections["batch"].shape == (num_boxes,)
    if num_boxes:
        assert detections["labels"].min().item() >= 0
        assert detections["labels"].max().item() < model.num_classes
        assert detections["batch"].min().item() >= 0
        assert detections["batch"].max().item() <= 1


def test_registered_weights_classes_match_num_classes() -> None:
    for entries in _REGISTERED_MODELS.values():
        for name, entry in entries.items():
            weights = entry["weights"]
            if weights is None or "classes" not in weights:
                continue

            num_classes = entry["hparams"].get("num_classes")
            if num_classes is not None:
                assert len(weights["classes"]) == num_classes, name


def test_registered_weights_urls_name_the_hub_repo() -> None:
    for entries in _REGISTERED_MODELS.values():
        for name, entry in entries.items():
            weights = entry["weights"]
            if weights is None:
                continue

            assert weights["url"] == f"hf://torch-pointcloud/{name}/resolve/main/model.safetensors", name
            assert weights["license"] in {"MIT", "Apache-2.0", "CC-BY-NC-4.0"}, name
            assert weights["author"], name
