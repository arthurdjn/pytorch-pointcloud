import pytest
import torch

from torch_pointcloud.utils.hilbert import binary2gray, decode, encode, gray2binary

HILBERT_2D_DEPTH1_PATH = [[0, 0], [0, 1], [1, 1], [1, 0]]
HILBERT_2D_DEPTH2_PATH = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 3],
    [1, 2],
    [2, 2],
    [2, 3],
    [3, 3],
    [3, 2],
    [3, 1],
    [2, 1],
    [2, 0],
    [3, 0],
]
HILBERT_3D_DEPTH1_PATH = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 0],
]
REFERENCE_3D_DEPTH4_LOCS = [[12, 15, 5], [0, 3, 11], [3, 7, 9], [3, 5, 2], [4, 7, 6]]
REFERENCE_3D_DEPTH4_CODES = [2396, 546, 1013, 102, 373]


def _assert_unit_step_path(path: torch.Tensor) -> None:
    """Every consecutive pair of cells differs by exactly one grid step along one axis."""
    steps = path[1:] - path[:-1]
    assert (steps.abs().sum(dim=1) == 1).all()


@pytest.mark.parametrize(
    "path,num_dims,num_bits",
    [
        pytest.param(HILBERT_2D_DEPTH1_PATH, 2, 1, id="2d-depth1"),
        pytest.param(HILBERT_2D_DEPTH2_PATH, 2, 2, id="2d-depth2"),
        pytest.param(HILBERT_3D_DEPTH1_PATH, 3, 1, id="3d-depth1"),
    ],
)
def test_encode_follows_known_curve(path: list, num_dims: int, num_bits: int) -> None:
    """Exhaustive known-answer check: the pinned path is a unit-step traversal of the full grid
    starting at the origin (i.e. a Hilbert curve), and `encode` maps cell $i$ of the path to code $i$."""
    locs = torch.tensor(path)
    assert locs.shape == (2 ** (num_dims * num_bits), num_dims)
    assert (locs[0] == 0).all()
    _assert_unit_step_path(locs)

    codes = encode(locs, num_dims=num_dims, num_bits=num_bits)
    assert torch.equal(codes, torch.arange(len(path)))


@pytest.mark.parametrize(
    "path,num_dims,num_bits",
    [
        pytest.param(HILBERT_2D_DEPTH1_PATH, 2, 1, id="2d-depth1"),
        pytest.param(HILBERT_2D_DEPTH2_PATH, 2, 2, id="2d-depth2"),
        pytest.param(HILBERT_3D_DEPTH1_PATH, 3, 1, id="3d-depth1"),
    ],
)
def test_decode_follows_known_curve(path: list, num_dims: int, num_bits: int) -> None:
    locs = decode(torch.arange(len(path)), num_dims=num_dims, num_bits=num_bits)
    assert torch.equal(locs, torch.tensor(path))


def test_encode_3d_depth4_reference_codes() -> None:
    codes = encode(torch.tensor(REFERENCE_3D_DEPTH4_LOCS), num_dims=3, num_bits=4)
    assert torch.equal(codes, torch.tensor(REFERENCE_3D_DEPTH4_CODES))


@pytest.mark.parametrize(
    "num_dims,num_bits",
    [
        pytest.param(2, 4, id="2d-depth4"),
        pytest.param(3, 4, id="3d-depth4"),
        pytest.param(3, 10, id="3d-depth10"),
        pytest.param(3, 16, id="3d-depth16"),
    ],
)
def test_encode_decode_roundtrip(num_dims: int, num_bits: int) -> None:
    g = torch.Generator().manual_seed(0)
    locs = torch.randint(0, 2**num_bits, (256, num_dims), generator=g)
    codes = encode(locs, num_dims=num_dims, num_bits=num_bits)
    assert codes.shape == (256,)
    assert codes.dtype == torch.int64
    assert torch.equal(decode(codes, num_dims=num_dims, num_bits=num_bits), locs)


def test_distinct_coords_give_distinct_codes() -> None:
    num_bits = 6
    g = torch.Generator().manual_seed(0)
    flat = torch.randperm(2 ** (3 * num_bits), generator=g)[:1024]
    side = 2**num_bits
    locs = torch.stack([flat // side**2, (flat // side) % side, flat % side], dim=1)
    codes = encode(locs, num_dims=3, num_bits=num_bits)
    assert codes.unique().numel() == locs.shape[0]


def test_gray_code_roundtrip() -> None:
    g = torch.Generator().manual_seed(0)
    bits = (torch.rand(64, 16, generator=g) > 0.5).byte()
    assert torch.equal(gray2binary(binary2gray(bits)), bits.bool())


def test_encode_rejects_mismatched_last_dim() -> None:
    with pytest.raises(ValueError, match="num_dims"):
        encode(torch.zeros(4, 2, dtype=torch.long), num_dims=3, num_bits=4)


def test_encode_rejects_too_many_bits() -> None:
    with pytest.raises(ValueError, match="int64"):
        encode(torch.zeros(4, 3, dtype=torch.long), num_dims=3, num_bits=22)


def test_decode_rejects_too_many_bits() -> None:
    with pytest.raises(ValueError, match="uint64"):
        decode(torch.zeros(4, dtype=torch.long), num_dims=3, num_bits=22)
