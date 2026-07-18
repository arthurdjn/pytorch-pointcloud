import pytest
from torch import Tensor

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


def test_detection_subclass_missing_decode_raises() -> None:
    class NoDecodeDetection(DetectionModel):
        def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
            return x

        def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
            return x

    with pytest.raises(TypeError, match="decode"):
        NoDecodeDetection(in_channels=3, num_classes=5)  # type: ignore[abstract]


def test_detection_subclass_missing_forward_features_raises() -> None:
    class NoBackboneDetection(DetectionModel):
        def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
            return x

        def decode(self, output: Tensor) -> Tensor:
            return output

    with pytest.raises(TypeError, match="forward_features"):
        NoBackboneDetection(in_channels=3, num_classes=5)  # type: ignore[abstract]


def test_detection_subclass_with_full_contract_instantiates() -> None:
    class MinimalDetection(DetectionModel):
        def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
            return x

        def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
            return x

        def decode(self, output: Tensor) -> Tensor:
            return output

    model = MinimalDetection(in_channels=3, num_classes=5)
    assert model.in_channels == 3
    assert model.num_classes == 5


def test_unimplemented_split_methods_raise_not_implemented() -> None:
    class ForwardOnlySegmentation(SegmentationModel):
        def forward(self, x: Tensor) -> Tensor:
            return x

    model = ForwardOnlySegmentation(in_channels=3, num_classes=5)
    with pytest.raises(NotImplementedError, match="forward_features"):
        model.forward_features()
    with pytest.raises(NotImplementedError, match="forward_decoder"):
        model.forward_decoder()
    with pytest.raises(NotImplementedError, match="forward_head"):
        model.forward_head()
    with pytest.raises(NotImplementedError, match="reset_classifier"):
        model.reset_classifier(num_classes=2)
