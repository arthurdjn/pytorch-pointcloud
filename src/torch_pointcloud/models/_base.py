from abc import ABCMeta
from typing import Any

import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels


class ClassificationModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def reset_classifier(self, num_classes: int, global_pool: Any) -> None: ...
    def forward_features(self, *_: Any, **__: Any) -> Any: ...
    def forward_head(self, *_: Any, **__: Any) -> Any: ...


class SegmentationModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def reset_classifier(self, num_classes: int) -> None: ...
    def forward_features(self, *_: Any, **__: Any) -> Any: ...
    def forward_decoder(self, *_: Any, **__: Any) -> Any: ...
    def forward_head(self, *_: Any, **__: Any) -> Any: ...


class DetectionModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def reset_classifier(self, num_classes: int) -> None: ...
    def forward_features(self, *_: Any, **__: Any) -> Any: ...
    def forward_head(self, *_: Any, **__: Any) -> Any: ...
