import torch
from torch.utils.data import Dataset

import torch_pointcloud.transforms as T
from torch_pointcloud.utils.data import PointCloudDataLoader, mix_pairs


def _scenes(sizes: list) -> list:
    return [{"pos": torch.rand(n, 3), "segment": torch.randint(0, 5, (n,))} for n in sizes]


def test_mix_pairs_merges_consecutive_samples() -> None:
    mix = T.Mix3D(keys=("pos", "segment"), instance_key=None, p=1.0)
    mixed = mix_pairs(_scenes([10, 20, 30, 40, 50]), mix, p=1.0)
    assert [s["pos"].shape[0] for s in mixed] == [30, 70, 50]
    assert all(s["segment"].shape[0] == s["pos"].shape[0] for s in mixed)


def test_mix_pairs_probability_and_small_batches() -> None:
    mix = T.Mix3D(keys=("pos", "segment"), instance_key=None, p=1.0)
    scenes = _scenes([10, 20])
    assert mix_pairs(scenes, mix, p=0.0) is scenes
    assert mix_pairs(scenes[:1], mix, p=1.0) is scenes[:1] or mix_pairs(scenes[:1], mix, p=1.0) == scenes[:1]
    generator = torch.Generator().manual_seed(0)
    outcomes = {len(mix_pairs(scenes, mix, p=0.5, generator=generator)) for _ in range(40)}
    assert outcomes == {1, 2}


class _Dataset(Dataset):
    def __init__(self) -> None:
        self.scenes = _scenes([10, 20, 30, 40])

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict:
        return self.scenes[index]


def test_dataloader_mixes_batches_before_collation() -> None:
    mix = T.Mix3D(keys=("pos", "segment"), instance_key=None, p=1.0)
    loader = PointCloudDataLoader(_Dataset(), batch_size=4, shuffle=False, mix=mix, mix_prob=1.0)
    batch = next(iter(loader))
    assert batch["pos"].shape[0] == 100
    assert batch["batch"].tolist() == [0] * 30 + [1] * 70
    plain = next(iter(PointCloudDataLoader(_Dataset(), batch_size=4, shuffle=False)))
    assert int(plain["batch"].max()) == 3
