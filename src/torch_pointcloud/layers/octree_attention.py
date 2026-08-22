"""OctFormer octree window attention with relative position encoding."""

import inspect
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_list_size
from torch_pointcloud.utils.imports import _OCNN_GITHUB_URL, optional_import
from torch_pointcloud.utils.ops import pad_tail

if TYPE_CHECKING:
    import ocnn
    from ocnn.octree import Octree
    from ocnn.octree import Octree as _OctreeTBase
else:
    _OctreeTBase = object

ocnn, _OCNN_AVAILABLE = optional_import("ocnn", url=_OCNN_GITHUB_URL)
Octree, _ = optional_import("ocnn.octree", "Octree", url=_OCNN_GITHUB_URL)


INVALID_MASK_VALUE = -1e3


def __getattr__(name: str) -> Type["OctreeT"]:
    # Pickle saves the concrete class by reference through this module attribute (its `__qualname__`),
    # so unpickling in a fresh process builds it on demand.
    if name == "_OctreeTConcrete":
        return _octree_t_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Creating the `Octree` subclass at module scope would resolve `ocnn` (and initialize CUDA) at import
# time, so the concrete class is built lazily on first instantiation.
@lru_cache
def _octree_t_cls() -> Type["OctreeT"]:
    if not _OCNN_AVAILABLE:
        raise ImportError(f"Optional module `ocnn` is required to use `OctreeT`. Install it from {_OCNN_GITHUB_URL}.")

    class _OctreeT(OctreeT, Octree):
        pass

    _OctreeT.__name__ = "OctreeT"
    _OctreeT.__qualname__ = "_OctreeTConcrete"
    return _OctreeT


class OctreeT(_OctreeTBase):
    r"""An enhanced Octree with transformer-specific capabilities (patching, dilation, masking).

    Can be instantiated directly like a standard Octree, or created from an existing
    Octree instance using `OctreeT.from_octree()`.

    Once a `OctreeT` is instantiated, you can build the transformer context
    (i.e. all attention masks and relative positions) by calling the method `construct_all_attention_context()`.

    Example:
        ```pycon
        >>> octree_t = OctreeT.from_octree(octree, patch_size=26, dilation=4)  # doctest: +SKIP
        >>> octree_t.construct_all_attention_context(  # doctest: +SKIP
        ...     nempty=True,
        ...     min_depth=6,
        ...     max_depth=10,
        ... )
        >>> octree_t.masks[6].shape  # doctest: +SKIP
        >>> octree_t.dilated_masks[6].shape  # doctest: +SKIP
        >>> octree_t.rel_pos[6].shape  # doctest: +SKIP
        >>> octree_t.dilated_rel_pos[6].shape  # doctest: +SKIP

        ```
    """

    __signature__: ClassVar[inspect.Signature]

    def __new__(cls, *args: Any, **kwargs: Any) -> "OctreeT":
        concrete = _octree_t_cls() if cls is OctreeT else cls
        return super().__new__(concrete)

    def __init__(
        self,
        depth: int,
        patch_size: int,
        dilation: int,
        full_depth: int = 2,
        batch_size: int = 1,
        device: Union[torch.device, str] = "cpu",
        **kwargs: Any,
    ):
        super().__init__(depth, full_depth, batch_size, device, **kwargs)
        self.patch_size = patch_size
        self.dilation = dilation
        self.invalid_mask_value = INVALID_MASK_VALUE

        self.masks: List[Optional[Tensor]]
        self.dilated_masks: List[Optional[Tensor]]
        self.rel_pos: List[Optional[Tensor]]
        self.dilated_rel_pos: List[Optional[Tensor]]
        self.nnum_t: Tensor
        self.nnum_a: Tensor

    @property
    def block_size(self) -> int:
        r"""Number of octree nodes a dilated patch spans, $\text{patch\_size} \cdot \text{dilation}$."""
        return self.patch_size * self.dilation

    def reset(self) -> None:
        r"""Resets the `OctreeT` to its initial state."""
        super().reset()
        self.invalid_mask_value = INVALID_MASK_VALUE

        num = self.depth + 1
        self.masks = ensure_list_size(None, num)
        self.dilated_masks = ensure_list_size(None, num)
        self.rel_pos = ensure_list_size(None, num)
        self.dilated_rel_pos = ensure_list_size(None, num)
        self.nnum_t = torch.zeros(num, device=self.device)
        self.nnum_a = torch.zeros(num, device=self.device)

    @classmethod
    def from_octree(cls, octree: Octree, patch_size: int, dilation: int, **kwargs: Any) -> "OctreeT":
        r"""Creates an `OctreeT` from an `Octree`.

        Args:
            octree: The `Octree` to create the `OctreeT` from.
            patch_size: The patch size to use for the `OctreeT`.
            dilation: The dilation to use for the `OctreeT`.
            **kwargs: Additional keyword arguments to pass to the `OctreeT` constructor.
        """
        out = cls(
            depth=octree.depth,
            full_depth=octree.full_depth,
            batch_size=octree.batch_size,
            device=octree.device,
            patch_size=patch_size,
            dilation=dilation,
            **kwargs,
        )
        out.__dict__.update(octree.__dict__)
        return out

    def construct_all_attention_context(
        self,
        nempty: bool = False,
        min_depth: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        r"""Constructs all attention context for the octree.

        Args:
            nempty: Whether to use non-empty nodes.
            min_depth: The start depth of the octree to construct the context for.
            max_depth: The end depth of the octree to construct the context for.
        """
        self.nnum_t = self.nnum_nempty if nempty else self.nnum
        self.nnum_a = ((self.nnum_t / self.block_size).ceil() * self.block_size).int()

        min_depth = min_depth or self.full_depth
        max_depth = max_depth or self.depth
        for d in range(min_depth, max_depth + 1):
            self.construct_attention_context(d, nempty)

    def construct_attention_context(self, depth: int, nempty: bool = False) -> None:
        r"""Calculates attention masks, relative positions, and padding indices
        required for attention operations.

        Args:
            depth: The depth of the octree to construct the context for.
            nempty: Whether to use non-empty nodes.
        """
        batch = self.batch_id(depth, nempty)
        padded_batch = self.pad_to_patch_size(batch, depth, fill_value=self.batch_size)

        self._construct_masks(padded_batch, depth)
        self._construct_rel_pos(depth, nempty)

    def _construct_masks(self, batch: Tensor, depth: int) -> None:
        mask = batch.view(-1, self.patch_size)
        self.masks[depth] = self._construct_mask(mask)

        mask = batch.view(-1, self.patch_size, self.dilation)
        mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
        self.dilated_masks[depth] = self._construct_mask(mask)

    def _construct_mask(self, mask: Tensor) -> Tensor:
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, self.invalid_mask_value)
        return attn_mask

    def _construct_rel_pos(self, depth: int, nempty: bool = False) -> None:
        key = self.key(depth, nempty)
        key = self.pad_to_patch_size(key, depth)
        x, y, z, _ = ocnn.octree.key2xyz(key, depth)
        xyz = torch.stack([x, y, z], dim=1)

        xyz_local = xyz.view(-1, self.patch_size, 3)
        self.rel_pos[depth] = xyz_local.unsqueeze(2) - xyz_local.unsqueeze(1)

        xyz = xyz.view(-1, self.patch_size, self.dilation, 3)
        xyz = xyz.transpose(1, 2).reshape(-1, self.patch_size, 3)
        self.dilated_rel_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

    def pad_to_patch_size(self, x: Tensor, depth: int, fill_value: float = 0) -> Tensor:
        r"""Pads `x` along its first dimension so the node count at `depth` is a whole number of patches."""
        pad_size = int(self.nnum_a[depth] - self.nnum_t[depth])
        return pad_tail(x, pad_size, fill_value=fill_value, dim=0)

    def unpad(self, x: Tensor, depth: int) -> Tensor:
        r"""Drops the padding added by `pad_to_patch_size`, restoring the real node count at `depth`."""
        original_size = self.nnum_t[depth]
        return x[:original_size]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"depth={self.depth}, full_depth={self.full_depth}, "
            f"patch_size={self.patch_size}, dilation={self.dilation}, "
            f"batch_size={self.batch_size}, device={self.device})"
        )


# `inspect.signature` reads a class's own `__new__` before `__init__`, which would report the lazy
# facade's opaque `(*args, **kwargs)`; pin the real constructor signature explicitly.
_init_signature = inspect.signature(OctreeT.__init__)
OctreeT.__signature__ = _init_signature.replace(parameters=list(_init_signature.parameters.values())[1:])


class RPE(nn.Module):
    r"""Relative Position Encoding (RPE) module used within the `OctreeAttention` module.

    Args:
        patch_size: The patch size to use for the RPE.
        num_heads: The number of heads to use for the RPE.
        dilation: The dilation to use for the RPE.
    """

    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = int(0.8 * patch_size * dilation**0.5)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def pos_to_idx(self, pos: Tensor) -> Tensor:
        r"""Clamps the relative positions to $\pm$ `pos_bnd` and maps each axis to its slice of the RPE table."""
        mul = torch.arange(3, device=pos.device) * self.rpe_num
        pos = pos.clamp(-self.pos_bnd, self.pos_bnd)
        idx = pos + (self.pos_bnd + mul)
        return idx

    def forward(self, pos: Tensor) -> Tensor:
        idx = self.pos_to_idx(pos)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, pos_bnd={self.pos_bnd}, dilation={self.dilation}"


class OctreeAttention(nn.Module):
    r"""Multi-head self-attention restricted to windows of `patch_size` consecutive octree nodes.

    A `dilation` above $1$ interleaves the nodes before windowing, so a patch spans a wider region at the
    same cost. The attention masks and relative positions come from the `OctreeT`, which must have been
    built with `construct_all_attention_context()`.

    Args:
        channels: Number of input and output channels.
        patch_size: Number of octree nodes attending to each other.
        num_heads: Number of attention heads.
        dilation: Stride between the nodes of a patch.
        qkv_bias: Whether to use a bias in the QKV projection.
        qk_scale: Scaling factor for the QK matrix. Defaults to the inverse square root of the head dimension.
        attn_drop: Dropout rate for the attention.
        proj_drop: Dropout rate for the output projection.
        use_rpe: Whether to add the relative position encoding to the attention logits.
    """

    def __init__(
        self,
        channels: int,
        patch_size: int,
        num_heads: int,
        dilation: int = 1,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rpe: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.use_rpe = use_rpe
        self.scale = qk_scale or (channels // num_heads) ** -0.5

        self.rpe = RPE(patch_size, num_heads, dilation) if use_rpe else None
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    @property
    def dilated(self) -> bool:
        r"""Whether the patches interleave nodes rather than taking them consecutively."""
        return self.dilation > 1

    def forward(self, x: Tensor, octree: OctreeT, depth: int) -> Tensor:
        x, rel_pos, mask = self._prepare_inputs(x, octree, depth)
        x = self.forward_attn(x, rel_pos, mask)
        x = self._postprocess_output(x, octree, depth)
        x = self.forward_proj(x)
        return x

    def forward_attn(self, x: Tensor, rel_pos: Tensor, mask: Tensor) -> Tensor:
        r"""Runs the windowed attention on patched features of shape $(N, K, C)$, returning $(N \cdot K, C)$."""
        H, K, C = self.num_heads, self.patch_size, self.channels

        qkv = self.qkv(x).reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N, H, K, C')
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)  # (N, H, K, K)
        attn = self.forward_rpe(attn, rel_pos)  # (N, H, K, K)
        attn = attn + mask.unsqueeze(1)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(-1, C)
        return x

    def forward_proj(self, x: Tensor) -> Tensor:
        r"""Applies the output projection and its dropout."""
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_rpe(self, attn: Tensor, rel_pos: Optional[Tensor] = None) -> Tensor:
        r"""Adds the relative position bias to the attention logits, or returns them unchanged when `use_rpe` is off."""
        if not self.use_rpe:
            return attn

        if self.rpe is None:
            raise ValueError("`rpe` must be set when `use_rpe` is True.")
        if rel_pos is None:
            raise ValueError("`rel_pos` must be provided when `use_rpe` is True")

        return attn + self.rpe(rel_pos)

    def _prepare_inputs(self, x: Tensor, octree: OctreeT, depth: int) -> Tuple[Tensor, Tensor, Tensor]:
        K, C, D = self.patch_size, self.channels, self.dilation

        x = octree.pad_to_patch_size(x, depth)
        rel_pos = octree.dilated_rel_pos[depth] if self.dilated else octree.rel_pos[depth]
        mask = octree.dilated_masks[depth] if self.dilated else octree.masks[depth]
        assert rel_pos is not None and mask is not None, (
            "`rel_pos` and `mask` must be set. Please ensure "
            "the `OctreeT` has been constructed with `construct_all_attention_context()`."
        )

        if self.dilated:
            x = x.view(-1, K, D, C).transpose(1, 2).reshape(-1, C)

        x = x.view(-1, K, C)
        return x, rel_pos, mask

    def _postprocess_output(self, x: Tensor, octree: OctreeT, depth: int) -> Tensor:
        K, C, D = self.patch_size, self.channels, self.dilation

        if self.dilated:
            x = x.view(-1, D, K, C).transpose(1, 2).reshape(-1, C)

        return octree.unpad(x, depth)
