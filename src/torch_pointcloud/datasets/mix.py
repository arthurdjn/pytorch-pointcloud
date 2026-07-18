"""Dataset wrapper that mixes each sample with a random partner via a pairwise transform."""

from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset, get_worker_info


class MixDataset(Dataset):
    """Draws a second random sample and applies a pairwise mix transform.

    Each source sample is produced by the wrapped dataset (its own `transform` runs first), then a
    partner sample is drawn at a random index and merged via `mix(data, other)`. Pairwise mixes such
    as `Mix3D`, `LaserMix`, and `PolarMix` fit this contract; the mix's own `p` decides how often the
    merge actually happens.

    Args:
        dataset: The wrapped dataset; its own `transform` runs per source sample before mixing.
        mix: Pairwise transform called as `mix(data, other)`.
        transform: Optional per-sample transform applied after mixing.
        generator: Optional `torch.Generator` for the partner index. Every `DataLoader` worker receives a
            replica of this generator, so inside a worker the partner indices are drawn from a dedicated
            stream seeded with the generator's initial seed plus the worker id.
    """

    def __init__(
        self,
        dataset: Dataset,
        mix: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.dataset = dataset
        self.mix = mix
        self.transform = transform
        self.generator = generator
        self._worker_generator: Optional[torch.Generator] = None

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.dataset[index]
        generator = self.generator
        worker_info = get_worker_info()
        if generator is not None and worker_info is not None:
            # Workers hold identical replicas of the generator; fold in the worker id so their streams differ.
            if self._worker_generator is None:
                self._worker_generator = torch.Generator()
                self._worker_generator.manual_seed(generator.initial_seed() + worker_info.id)
            generator = self._worker_generator

        other_index = int(torch.randint(len(self.dataset), (1,), generator=generator).item())  # type: ignore[arg-type]
        other = self.dataset[other_index]
        mixed = self.mix(data, other)
        return self.transform(mixed) if self.transform is not None else mixed
