#type: ignore
from ._registry import register_model


@register_model
def resnet():
    return "OK"