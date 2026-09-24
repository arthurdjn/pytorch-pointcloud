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

from .augmentation import *
from .base import *
from .box import *
from .geometry import *
from .masking import *
from .mixing import *
from .octree import *
from .sampling import *
from .scaling import *
from .utils import *
from .voxelization import *
