import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models import ClassificationModel, DetectionModel, SegmentationModel


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(ClassificationModel, id="classification"),
        pytest.param(SegmentationModel, id="segmentation"),
        pytest.param(DetectionModel, id="detection"),
    ],
)
def test_task_abc_cannot_be_instantiated(model_cls: type) -> None:
    with pytest.raises(TypeError, match="abstract"):
        model_cls(in_channels=3, num_classes=5)


def test_classification_subclass_missing_forward_raises() -> None:
    class NoForwardClassification(ClassificationModel):
        pass

    with pytest.raises(TypeError, match="forward"):
        NoForwardClassification(in_channels=3, num_classes=5)  # type: ignore[abstract]


def test_segmentation_subclass_missing_forward_raises() -> None:
    class NoForwardSegmentation(SegmentationModel):
        pass

    with pytest.raises(TypeError, match="forward"):
        NoForwardSegmentation(in_channels=3, num_classes=5)  # type: ignore[abstract]


@pytest.mark.parametrize("method", ["num_features", "forward_features", "forward_head", "decode"])
def test_detection_subclass_missing_member_raises(method: str) -> None:
    class ForwardOnlyDetection(DetectionModel):
        def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
            return x

    with pytest.raises(TypeError, match=method):
        ForwardOnlyDetection(in_channels=3, num_classes=5)  # type: ignore[abstract]


def test_detection_subclass_with_full_contract_instantiates() -> None:
    class MinimalDetection(DetectionModel):
        @property
        def num_features(self) -> int:
            return self.in_channels

        def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
            return x

        def forward_head(self, features: Tensor) -> Tensor:
            return features

        def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
            features = self.forward_features(x, pos, batch)
            return self.forward_head(features)

        def decode(self, output: Tensor) -> Tensor:
            return output

    model = MinimalDetection(in_channels=3, num_classes=5)
    assert model.in_channels == 3
    assert model.num_classes == 5
    assert model.num_features == 3


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(ClassificationModel, id="classification"),
        pytest.param(SegmentationModel, id="segmentation"),
    ],
)
@pytest.mark.parametrize(
    "method", ["num_features", "configure_head", "reset_classifier", "forward_features", "forward_head"]
)
def test_subclass_missing_split_method_raises(model_cls: type, method: str) -> None:
    class ForwardOnly(model_cls):
        def forward(self, x: Tensor) -> Tensor:
            return x

    with pytest.raises(TypeError, match=method):
        ForwardOnly(in_channels=3, num_classes=5)


def test_segmentation_subclass_with_full_contract_instantiates() -> None:
    class MinimalSegmentation(SegmentationModel):
        def __init__(self, in_channels: int, num_classes: int) -> None:
            super().__init__(in_channels=in_channels, num_classes=num_classes)
            self.head = self.configure_head()

        @property
        def num_features(self) -> int:
            return self.in_channels

        def configure_head(self) -> nn.Module:
            return nn.Identity() if self.num_classes == 0 else nn.Linear(self.num_features, self.num_classes)

        def reset_classifier(self, num_classes: int) -> None:
            self.num_classes = num_classes
            self.head = self.configure_head()

        def forward_features(self, x: Tensor) -> Tensor:
            return x

        def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
            return x if pre_logits else self.head(x)

        def forward(self, x: Tensor) -> Tensor:
            features = self.forward_features(x)
            return self.forward_head(features)

    model = MinimalSegmentation(in_channels=3, num_classes=5)
    assert model.num_features == 3
    with pytest.raises(NotImplementedError, match="forward_decoder"):
        model.forward_decoder()

    model.reset_classifier(0)
    assert isinstance(model.head, nn.Identity)
    x = torch.randn(4, 3)
    assert model(x).shape == (4, model.num_features)
