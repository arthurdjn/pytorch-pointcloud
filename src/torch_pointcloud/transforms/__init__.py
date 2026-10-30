r"""Dict-based transforms and their tensor-level functional equivalents.

All transforms in this module operate on a **single scene** (one sample, pre-collate).
The `batch` key is not consumed at this layer; reductions like `mean`, `min`, and `max`
are computed over the whole tensor, not per-batch. Apply transforms before DataLoader
collation if you want per-scene behavior.

Transforms are non-mutating: each transform returns a new shallow-copy dict. Tensors
inside the dict are not cloned unless the transform's documentation says so.

Sampling keys: a transform that changes the number of points can record how to get back to its input.
Voxelizers (`Voxelize`, `DivisiblePad`) write `inverse`, $(N_\text{in},)$, input row to output row, under
`dst_inverse_key`, so `preds[inverse]` scores at input resolution. Selection samplers (`RandomSample`,
`FarthestPointSample`, `SphereCrop`, `RemoveNearOrigin`, `RandomDropout`, `ShufflePoint`, `ApplyMask`, `Slice`
along `dim=0`) write `index`, $(N_\text{out},)$, output row to input row, under `dst_index_key`. Both compose
through a prior value at the same key, so a chain of samplers of one kind stays addressed to the outermost
input. Registered evaluation pipelines set these keys and copy the source cloud to `origin_pos` /
`origin_segment` with `CopyItems` before their first sampling step.
"""

from .augmentation import (
    RandomColorAutoContrast,
    RandomColorDrop,
    RandomColorGrayScale,
    RandomColorJitter,
    RandomColorShift,
    RandomElasticDistortion,
    RandomFlip,
    RandomJitter,
    RandomLinearJitter,
    RandomRotate,
    RandomRotateChoice,
    RandomScale,
    RandomTranslate,
)
from .base import (
    Compose,
    DictTransform,
    RandomApply,
    Transform,
)
from .box import (
    InstanceToBox,
    PasteBoxes,
    RelabelBoxes,
)
from .geometry import (
    AxisMinOffset,
    BBoxCenter,
    EstimateNormals,
    Quantize,
    Shift,
    Translate,
)
from .masking import (
    ApplyMask,
    BoxMask,
    CubeMask,
    LabelMask,
    MaskedFill,
    RemoveNearOrigin,
    SphereMask,
)
from .mixing import (
    LaserMix,
    Mix3D,
    PolarMix,
)
from .octree import (
    BuildOctree,
    OctreeFeatures,
)
from .sampling import (
    FarthestPointSample,
    RandomBlockCrop,
    RandomDropout,
    RandomSample,
    RandomSampleFaceVertices,
    ShufflePoint,
    Slice,
    SphereCrop,
)
from .scaling import (
    MaxAbsScale,
    Normalize,
    Rescale,
    Standardize,
)
from .utils import (
    Abs,
    Add,
    Cat,
    Clamp,
    CopyItems,
    Divide,
    DivideItems,
    KeepItems,
    Log,
    NormalizedDifference,
    OneHot,
    OnesLike,
    Reduce,
    Relabel,
    RenameItems,
    Scale,
    SetValue,
    SubtractItems,
    ToDevice,
    ToFloat,
    ToTensor,
)
from .voxelization import (
    DivisiblePad,
    HardVoxelize,
    Voxelize,
)

__all__ = [
    "RandomColorAutoContrast",
    "RandomColorDrop",
    "RandomColorGrayScale",
    "RandomColorJitter",
    "RandomColorShift",
    "RandomElasticDistortion",
    "RandomFlip",
    "RandomJitter",
    "RandomLinearJitter",
    "RandomRotate",
    "RandomRotateChoice",
    "RandomScale",
    "RandomTranslate",
    "Compose",
    "DictTransform",
    "RandomApply",
    "Transform",
    "InstanceToBox",
    "PasteBoxes",
    "RelabelBoxes",
    "AxisMinOffset",
    "BBoxCenter",
    "EstimateNormals",
    "Quantize",
    "Shift",
    "Translate",
    "ApplyMask",
    "BoxMask",
    "CubeMask",
    "LabelMask",
    "MaskedFill",
    "RemoveNearOrigin",
    "SphereMask",
    "LaserMix",
    "Mix3D",
    "PolarMix",
    "BuildOctree",
    "OctreeFeatures",
    "FarthestPointSample",
    "RandomBlockCrop",
    "RandomDropout",
    "RandomSample",
    "RandomSampleFaceVertices",
    "ShufflePoint",
    "Slice",
    "SphereCrop",
    "MaxAbsScale",
    "Normalize",
    "Rescale",
    "Standardize",
    "Abs",
    "Add",
    "Cat",
    "Clamp",
    "CopyItems",
    "Divide",
    "DivideItems",
    "KeepItems",
    "Log",
    "NormalizedDifference",
    "OneHot",
    "OnesLike",
    "Reduce",
    "Relabel",
    "RenameItems",
    "Scale",
    "SetValue",
    "SubtractItems",
    "ToDevice",
    "ToFloat",
    "ToTensor",
    "DivisiblePad",
    "HardVoxelize",
    "Voxelize",
]
