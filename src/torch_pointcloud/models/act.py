"""ACT classification and pretraining models.

{{ paper("2212.08320") }}

ACT (Autoencoders as Cross-Modal Teachers) pretrains the Point-MAE transformer by distilling a discrete variational
autoencoder whose tokens pass through a frozen, prompted 2D image transformer; that teacher only exists at
pretraining time, so the released fine-tuned classifiers are `PointMAEClassification` networks and load as such.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import PointPatchEmbed, TransformerBlock
from torch_pointcloud.ops.cluster import group
from torch_pointcloud.ops.masking import random_mask
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import PretrainingModel
from ._registry import WeightsDict, register_model
from .point_bert import FoldingDecoder, TokenDGCNN
from .point_mae import PointMAEClassification, TransformerDecoder, TransformerEncoder


class ACTDiscreteVAE(PretrainingModel):
    r"""ACT discrete VAE with a prompted image transformer, as in :arxiv: [Autoencoders as Cross-Modal Teachers: Can
    Pretrained 2D Image Transformers Help 3D Representation Learning?](https://arxiv.org/abs/2212.08320), adapted from
    :github: [RunpeiDong/ACT](https://github.com/RunpeiDong/ACT).

    The Point-BERT dVAE (a mini-PointNet patch encoder, a DGCNN mapping patches to codebook logits, a codebook, a
    second DGCNN and a folding decoder) with a 2D image transformer between the codebook and the second DGCNN. The
    sampled codebook embeddings are projected to the image width, positioned by an MLP of the patch centers, and run
    through the frozen transformer blocks behind `num_prompt_tokens` learnable prompt tokens, replaced by a fresh set
    before every block after the first; the prompts are dropped at the output, which is projected back to
    `tokens_dims`. `group_features` returns the features that ACT pretraining distills, `forward` the patch
    reconstruction.

    Args:
        in_channels: Number of input channels ($0$, coordinates only).
        num_groups: Number of patches $G$.
        group_size: Number of points $M$ per patch.
        encoder_dims: Width of the mini-PointNet patch embedding.
        token_local_channels: Hidden widths of the patch embedder's per-point MLP.
        token_global_channels: Hidden widths of the patch embedder's per-patch MLP.
        num_tokens: Codebook vocabulary size.
        tokens_dims: Codebook embedding width.
        decoder_dims: Width fed to the folding decoder.
        image_dim: Width of the image transformer.
        image_depth: Number of image transformer blocks.
        image_num_heads: Number of attention heads of the image transformer.
        num_prompt_tokens: Number of prompt tokens per image transformer block.
        prompt_dropout: Dropout applied to the prompt tokens.
        act: Activation of the patch encoder and the folding decoder.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization of the patch encoder and the folding decoder.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: a dict with `logits` $(B, G, \text{num\_tokens})$, the patches and their reconstructions.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.act import ACTDiscreteVAE

        model = ACTDiscreteVAE(in_channels=0, image_dim=96, image_depth=2, image_num_heads=2, num_prompt_tokens=4)
        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        out = model(None, pos, batch)
        print(out["logits"].shape, out["fine"].shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        *,
        image_dim: int,
        image_depth: int,
        image_num_heads: int,
        num_prompt_tokens: int,
        num_groups: int = 64,
        group_size: int = 32,
        encoder_dims: int = 384,
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        num_tokens: int = 8192,
        tokens_dims: int = 384,
        decoder_dims: int = 384,
        prompt_dropout: float = 0.1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels)
        self.num_groups = num_groups
        self.group_size = group_size
        self.encoder_dims = encoder_dims
        self.num_tokens = num_tokens
        self.tokens_dims = tokens_dims
        self.decoder_dims = decoder_dims
        self.image_dim = image_dim
        self.image_depth = image_depth
        self.num_prompt_tokens = num_prompt_tokens

        self.encoder = PointPatchEmbed(
            embed_dim=encoder_dims,
            local_channels=token_local_channels,
            global_channels=token_global_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.dgcnn_1 = TokenDGCNN(in_channels=encoder_dims, out_channels=num_tokens)
        self.codebook = nn.Parameter(torch.zeros(num_tokens, tokens_dims))
        self.image_proj_in = nn.Linear(tokens_dims, image_dim)
        self.image_pos_embed = MLP([3, 128, image_dim], act="gelu", norm=None, plain_last=True)
        self.prompt_token = nn.Parameter(torch.zeros(1, num_prompt_tokens, image_dim))
        self.prompt_pos = nn.Parameter(torch.randn(1, num_prompt_tokens, image_dim))
        self.deep_prompt_token = nn.Parameter(torch.zeros(image_depth - 1, num_prompt_tokens, image_dim))
        self.deep_prompt_pos = nn.Parameter(torch.randn(image_depth - 1, num_prompt_tokens, image_dim))
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.image_blocks = nn.ModuleList(
            [
                TransformerBlock(image_dim, image_num_heads, qkv_bias=True, norm_kwargs=dict(eps=1e-6))
                for _ in range(image_depth)
            ]
        )
        self.image_norm = nn.LayerNorm(image_dim, eps=1e-6)
        self.image_proj_out = nn.Linear(image_dim, tokens_dims)
        self.dgcnn_2 = TokenDGCNN(in_channels=tokens_dims, out_channels=decoder_dims)
        self.decoder = FoldingDecoder(
            in_channels=decoder_dims,
            num_fine=group_size,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.image_blocks.requires_grad_(False)
        self.image_norm.requires_grad_(False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in (self.prompt_token, self.prompt_pos, self.deep_prompt_token, self.deep_prompt_pos):
            nn.init.trunc_normal_(param, std=0.02)

    def embed_image(self, x: Tensor, center: Tensor) -> Tensor:
        r"""Run sampled codebook embeddings through the prompted image transformer.

        Args:
            x: Codebook embeddings of shape $(B, G, \text{tokens\_dims})$.
            center: Patch centers of shape $(B, G, 3)$.

        Returns:
            The embeddings mapped back to $(B, G, \text{tokens\_dims})$.
        """
        B = x.shape[0]
        P = self.num_prompt_tokens
        x = torch.cat([self.prompt_dropout(self.prompt_token.expand(B, -1, -1)), self.image_proj_in(x)], dim=1)
        pos = torch.cat([self.prompt_pos.expand(B, -1, -1), self.image_pos_embed(center)], dim=1)
        for i, block in enumerate(self.image_blocks):
            if i > 0:
                prompt = self.prompt_dropout(self.deep_prompt_token[i - 1].expand(B, -1, -1))
                x = torch.cat([prompt, x[:, P:]], dim=1)
                pos = torch.cat([self.deep_prompt_pos[i - 1].expand(B, -1, -1), pos[:, P:]], dim=1)
            x = block(x + pos)
        return self.image_proj_out(self.image_norm(x)[:, P:])

    def group_features(self, neighborhood: Tensor, center: Tensor, temperature: float = 1.0) -> Tensor:
        r"""Return the patch features of the teacher: hard-sampled codebook entries through the image transformer
        and the second DGCNN.

        Args:
            neighborhood: Centered patches of shape $(B, G, M, 3)$.
            center: Patch centers of shape $(B, G, 3)$.
            temperature: Temperature of the Gumbel-softmax sampling of the codebook.

        Returns:
            Patch features of shape $(B, G, \text{decoder\_dims})$.
        """
        logits = self.dgcnn_1(self.encoder(neighborhood), center)
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=True)
        sampled = torch.einsum("bgn,nc->bgc", soft_one_hot, self.codebook)
        return self.dgcnn_2(self.embed_image(sampled, center), center)

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, Tensor]:
        neighborhood, center = group(pos, batch, self.num_groups, self.group_size, random_start=self.training)
        logits = self.dgcnn_1(self.encoder(neighborhood), center)
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)
        sampled = torch.einsum("bgn,nc->bgc", soft_one_hot, self.codebook)
        feature = self.dgcnn_2(self.embed_image(sampled, center), center)
        coarse, fine = self.decoder(feature)
        return {
            "logits": logits,
            "neighborhood": neighborhood,
            "center": center,
            "coarse": coarse,
            "fine": fine,
        }


class ACTPretraining(PretrainingModel):
    r"""ACT feature-distillation pretraining model, as in :arxiv: [Autoencoders as Cross-Modal Teachers: Can
    Pretrained 2D Image Transformers Help 3D Representation Learning?](https://arxiv.org/abs/2212.08320), adapted from
    :github: [RunpeiDong/ACT](https://github.com/RunpeiDong/ACT).

    Masks a fraction of the patch tokens and encodes the visible ones, behind a class token, with the Point-MAE
    transformer; a transformer decoder fills learnable mask tokens at the masked centers and a linear projection maps
    them to the patch features of a frozen `ACTDiscreteVAE` teacher. `forward` returns the student predictions and
    the teacher targets of the masked patches, for a cosine-similarity objective.

    Args:
        in_channels: Number of input channels (unused beyond coordinates; kept for the registry contract).
        image_dim: Width of the teacher's image transformer.
        image_depth: Number of blocks of the teacher's image transformer.
        image_num_heads: Number of attention heads of the teacher's image transformer.
        num_prompt_tokens: Number of prompt tokens per block of the teacher's image transformer.
        embed_dim: Token channels.
        encoder_depth: Number of encoder transformer blocks.
        decoder_depth: Number of decoder transformer blocks.
        num_heads: Number of encoder attention heads.
        decoder_num_heads: Number of decoder attention heads.
        num_groups: Number of groups (FPS centers) per sample.
        group_size: Number of neighbors per group.
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-group MLP.
        pos_embed_channels: Hidden widths of the encoder and decoder positional-embedding MLPs.
        num_tokens: Codebook vocabulary size of the teacher.
        tokens_dims: Width of the teacher's codebook embeddings and patch features.
        mask_ratio: Fraction of tokens to mask.
        drop_path: Stochastic-depth drop-path rate.
        act: Activation function of the transformers.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization of the transformers.
        norm_kwargs: Extra arguments for the normalization.
        spatial_dim: Spatial dimension of the input point cloud.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: predictions and targets, each of shape $(B \cdot G_\text{mask}, \text{tokens\_dims})$.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        image_dim: int,
        image_depth: int,
        image_num_heads: int,
        num_prompt_tokens: int,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        decoder_depth: int = 2,
        num_heads: int = 6,
        decoder_num_heads: int = 6,
        num_groups: int = 64,
        group_size: int = 32,
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        num_tokens: int = 8192,
        tokens_dims: int = 384,
        mask_ratio: float = 0.8,
        drop_path: float = 0.1,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        spatial_dim: int = 3,
    ) -> None:
        super().__init__(in_channels=in_channels)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"`mask_ratio` must be in (0, 1), got {mask_ratio}.")

        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.decoder_num_heads = decoder_num_heads
        self.num_groups = num_groups
        self.group_size = group_size
        self.pos_embed_channels = pos_embed_channels
        self.mask_ratio = mask_ratio
        self.drop_path = drop_path
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.spatial_dim = spatial_dim

        self.encoder = PointPatchEmbed(
            embed_dim=embed_dim,
            local_channels=encoder_local_channels,
            global_channels=encoder_global_channels,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = self.configure_pos_embed()
        self.blocks = self.configure_blocks()
        self.norm_f = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = self.configure_pos_embed()
        self.decoder = self.configure_decoder()
        self.proj_head = nn.Linear(embed_dim, tokens_dims)
        self.teacher = ACTDiscreteVAE(
            in_channels=0,
            image_dim=image_dim,
            image_depth=image_depth,
            image_num_heads=image_num_heads,
            num_prompt_tokens=num_prompt_tokens,
            num_groups=num_groups,
            group_size=group_size,
            num_tokens=num_tokens,
            tokens_dims=tokens_dims,
            decoder_dims=tokens_dims,
        )
        self.teacher.requires_grad_(False)
        self.reset_parameters()

    def configure_pos_embed(self) -> MLP:
        """Build a positional-embedding MLP mapping group centers to token channels."""
        return MLP(
            [self.spatial_dim, *self.pos_embed_channels, self.embed_dim],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=None,
            plain_last=True,
        )

    def configure_blocks(self) -> TransformerEncoder:
        """Build the transformer encoder with a linearly scaled stochastic-depth schedule."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.encoder_depth)]
        return TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.encoder_depth,
            drop_path=dpr,
            num_heads=self.num_heads,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder(self) -> TransformerDecoder:
        """Build the transformer decoder filling the masked tokens."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.decoder_depth)]
        return TransformerDecoder(
            embed_dim=self.embed_dim,
            depth=self.decoder_depth,
            drop_path=dpr,
            num_heads=self.decoder_num_heads,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def reset_parameters(self) -> None:
        for param in (self.cls_token, self.cls_pos, self.mask_token):
            nn.init.trunc_normal_(param, std=0.02)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        neighborhood, center = group(pos, batch, self.num_groups, self.group_size, random_start=self.training)
        B, G, _ = center.shape
        mask = random_mask(B, G, self.mask_ratio, device=center.device)

        tokens = self.encoder(neighborhood)
        C = tokens.shape[-1]
        x_vis = torch.cat([self.cls_token.expand(B, -1, -1), tokens[~mask].reshape(B, -1, C)], dim=1)
        pos_vis = self.pos_embed(center[~mask].reshape(B, -1, self.spatial_dim))
        x_vis = self.norm_f(self.blocks(x_vis, torch.cat([self.cls_pos.expand(B, -1, -1), pos_vis], dim=1)))[:, 1:]

        decoder_pos_vis = self.decoder_pos_embed(center[~mask].reshape(B, -1, self.spatial_dim))
        decoder_pos_mask = self.decoder_pos_embed(center[mask].reshape(B, -1, self.spatial_dim))
        M = decoder_pos_mask.shape[1]
        x_full = torch.cat([x_vis, self.mask_token.expand(B, M, -1)], dim=1)
        x_rec = self.decoder(x_full, torch.cat([decoder_pos_vis, decoder_pos_mask], dim=1), M)
        predictions = self.proj_head(x_rec).reshape(B * M, -1)

        with torch.no_grad():
            targets = self.teacher.group_features(neighborhood, center)[mask]
        return predictions, targets


@register_model(
    "act.modelnet40.runpei-dong",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/act.modelnet40.runpei-dong/resolve/0d58e7d65cbd4e3c793c6b212796bed5149f51e4/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 93.19, "mAcc": 89.90},
        classes=MODELNET40_CLASSES,
        author="runpei-dong",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
                allow_missing_keys=True,
            ),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
        ]
    ),
    hparams=dict(
        in_channels=0,
        num_classes=40,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=64,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def act_modelnet40_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "act.scanobjectnn-objbg.runpei-dong",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/act.scanobjectnn-objbg.runpei-dong/resolve/a5e3549e374ae49af099fd80920abc7b919de183/model.safetensors",
        dataset="scanobjectnn-objbg",
        metrics={"OA": 93.29, "mAcc": 91.48},
        classes=SCANOBJECTNN_CLASSES,
        author="runpei-dong",
        license="MIT",
    ),
    # The reference feeds the raw 2048-point ScanObjectNN sample, without sampling or normalization.
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=128,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def act_scanobjectnn_objbg_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "act.scanobjectnn-objonly.runpei-dong",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/act.scanobjectnn-objonly.runpei-dong/resolve/f7a3234ac7c1e0c22521bd91174cc35f68204438/model.safetensors",
        dataset="scanobjectnn-objonly",
        metrics={"OA": 91.22, "mAcc": 90.26},
        classes=SCANOBJECTNN_CLASSES,
        author="runpei-dong",
        license="MIT",
    ),
    # The reference feeds the raw 2048-point ScanObjectNN sample, without sampling or normalization.
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=128,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def act_scanobjectnn_objonly_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "act.scanobjectnn-hardest.runpei-dong",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/act.scanobjectnn-hardest.runpei-dong/resolve/c2d08b53f068e6b45453b8be437332ea9404cafa/model.safetensors",
        dataset="scanobjectnn-hardest",
        metrics={"OA": 88.27, "mAcc": 86.63},
        classes=SCANOBJECTNN_CLASSES,
        author="runpei-dong",
        license="MIT",
    ),
    # The reference feeds the raw 2048-point ScanObjectNN sample, without sampling or normalization.
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=128,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def act_scanobjectnn_hardest_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "act.dvae.runpei-dong",
    task="pretraining",
    weights=WeightsDict(
        url="hf://torch-pointcloud/act.dvae.runpei-dong/resolve/321e532150c191521fa4c91cec397d9077b04508/model.safetensors",
        dataset="shapenet55",
        author="runpei-dong",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        image_dim=768,
        image_depth=12,
        image_num_heads=12,
        num_prompt_tokens=64,
        num_groups=64,
        group_size=32,
        encoder_dims=384,
        token_local_channels=(128, 256),
        token_global_channels=(512,),
        num_tokens=8192,
        tokens_dims=384,
        decoder_dims=384,
        prompt_dropout=0.1,
    ),
)
def act_dvae(**kwargs: Any) -> ACTDiscreteVAE:
    return ACTDiscreteVAE(**kwargs)


@register_model(
    "act.pretrain.runpei-dong",
    task="pretraining",
    weights=WeightsDict(
        url="hf://torch-pointcloud/act.pretrain.runpei-dong/resolve/5b0e013b17fa7a1ca3042045e568a906aaa57932/model.safetensors",
        dataset="shapenet55",
        author="runpei-dong",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        image_dim=768,
        image_depth=12,
        image_num_heads=12,
        num_prompt_tokens=64,
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=2,
        num_heads=6,
        decoder_num_heads=6,
        num_groups=64,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        num_tokens=8192,
        tokens_dims=384,
        mask_ratio=0.8,
        drop_path=0.1,
        act="gelu",
        spatial_dim=3,
    ),
)
def act_pretrain(**kwargs: Any) -> ACTPretraining:
    return ACTPretraining(**kwargs)
