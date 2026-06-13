import pytest
import torch

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.xcube import XCubeDiffusion, XCubeVAE, fourier_encode, timestep_encoding
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _FVDB_AVAILABLE

DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"

_FULL_STACK = _FVDB_AVAILABLE and _CUDA_AVAILABLE


def test_fourier_encode_shape_and_order() -> None:
    x = torch.tensor([[0.5, -0.25, 1.0]])
    out = fourier_encode(x, num_freqs=5)
    assert out.shape == (1, 3 * (2 * 5 + 1))
    assert torch.equal(out[:, :3], x)
    assert torch.allclose(out[:, 3:6], torch.sin(x))
    assert torch.allclose(out[:, 6:9], torch.cos(x))
    assert torch.allclose(out[:, 9:12], torch.sin(2 * x))


def test_timestep_encoding_shape() -> None:
    emb = timestep_encoding(torch.tensor([0, 500, 999]), dim=64)
    assert emb.shape == (3, 64)
    assert torch.allclose(emb[0, :32], torch.ones(32))
    assert torch.allclose(emb[0, 32:], torch.zeros(32))


def test_xcube_registered_variants() -> None:
    names = list_models("xcube*", task="base")
    assert len(names) == 12
    for category in ("chair", "car", "plane"):
        for variant in ("vae-coarse", "vae-fine", "diffusion-coarse", "diffusion-fine"):
            assert f"xcube-{variant}-nvidia.shapenet-{category}" in names


@pytest.mark.skipif(not _FVDB_AVAILABLE, reason="fvdb is not installed")
def test_xcube_vae_create_model_hparams() -> None:
    model = create_model("xcube-vae-coarse-nvidia.shapenet-chair", task="base")
    assert isinstance(model, XCubeVAE)
    assert model.latent_channels == 16
    assert model.num_levels == 4
    assert model.voxel_size == 0.01
    fine = create_model("xcube-vae-fine-nvidia.shapenet-chair", task="base")
    assert isinstance(fine, XCubeVAE)
    assert fine.latent_channels == 8
    assert fine.num_levels == 3
    assert fine.unet.neck_bound is None


def _make_inputs(num_points: int = 2000, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    pos = (torch.rand(num_points * batch_size, 3) - 0.5) * 1.2
    batch = torch.repeat_interleave(torch.arange(batch_size), num_points)
    normal = torch.nn.functional.normalize(torch.randn(num_points * batch_size, 3), dim=1)
    return pos.to(DEVICE), batch.to(DEVICE), normal.to(DEVICE)


@pytest.mark.skipif(not _FULL_STACK, reason="fvdb or CUDA is not available")
def test_xcube_vae_forward() -> None:
    model = create_model("xcube-vae-coarse-nvidia.shapenet-chair", task="base").to(DEVICE).eval()
    pos, batch, normal = _make_inputs()
    with torch.no_grad():
        out = model(pos, batch, normal=normal)
    assert out["mu"].shape == (out["latent_grid"].total_voxels, 16)
    assert out["latent_grid"].total_voxels == 2 * 16**3
    assert sorted(out["structure_logits"].keys()) == [0, 1, 2, 3]
    for depth, logits in out["structure_logits"].items():
        assert logits.shape == (out["structure_logit_grids"][depth].total_voxels, 2)
    assert out["x"].shape[0] == out["grid"].total_voxels
    assert out["normal"].shape == (out["grid"].total_voxels, 3)


@pytest.mark.skipif(not _FULL_STACK, reason="fvdb or CUDA is not available")
def test_xcube_vae_encode_decode_round_trip_shapes() -> None:
    model = create_model("xcube-vae-fine-nvidia.shapenet-chair", task="base")
    assert isinstance(model, XCubeVAE)
    model = model.to(DEVICE).eval()
    pos, batch, normal = _make_inputs(num_points=500, batch_size=1)
    with torch.no_grad():
        z, grid = model.encode(pos, batch, normal=normal)
        out = model.decode(z, grid)
    assert z.shape == (grid.total_voxels, 8)
    assert sorted(out["structure_grids"].keys()) == [0, 1, 2]


@pytest.mark.skipif(not _FULL_STACK, reason="fvdb or CUDA is not available")
@pytest.mark.parametrize("dense", [True, False])
def test_xcube_diffusion_sample_smoke(dense: bool) -> None:
    vae = XCubeVAE(
        encoder_channels=8,
        channels=(8, 16),
        latent_channels=4,
        voxel_size=0.05,
        neck_bound=(4, 4, 4) if dense else None,
        use_normal=True,
        norm_kwargs={"num_groups": 2},
        with_normal_head=True,
    )
    model = XCubeDiffusion(
        vae=vae,
        model_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(2,),
        num_heads=2,
        dense=dense,
        latent_size=8,
        pos_embed_ijk=not dense,
        normal_cond=not dense,
        norm_kwargs={"num_groups": 2},
    )
    model = model.to(DEVICE).eval()

    if dense:
        out = model.sample(batch_size=2, num_steps=2)
    else:
        grid = model.latent_grid(2, DEVICE)
        normal = torch.randn(grid.total_voxels, 3, device=DEVICE)
        out = model.sample(grid=grid, normal=normal, num_steps=2)
    assert out["grid"].grid_count == 2
    assert out["x"].shape[0] == out["grid"].total_voxels


@pytest.mark.skipif(not _FULL_STACK, reason="fvdb or CUDA is not available")
def test_xcube_diffusion_training_forward() -> None:
    vae = XCubeVAE(
        encoder_channels=8,
        channels=(8, 16),
        latent_channels=4,
        voxel_size=0.05,
        neck_bound=(4, 4, 4),
        use_normal=True,
        norm_kwargs={"num_groups": 2},
    )
    model = XCubeDiffusion(
        vae=vae,
        model_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        dense=True,
        latent_size=8,
        norm_kwargs={"num_groups": 2},
    ).to(DEVICE)
    pos, batch, normal = _make_inputs(num_points=300, batch_size=2)
    out = model(pos, batch, normal=normal)
    assert out["pred"].shape == out["target"].shape
    assert out["pred"].shape[1] == 4
