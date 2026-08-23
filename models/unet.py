# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""
Modified from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
"""

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nn import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


class ConstantEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embedding_table = nn.Parameter(torch.empty((1, out_channels)))
        nn.init.uniform_(
            self.embedding_table, -(in_channels**0.5), in_channels**0.5
        )

    def forward(self, emb):
        return self.embedding_table.repeat(emb.shape[0], 1)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        emb_off=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if emb_off:
            self.emb_layers = ConstantEmbedding(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            )
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels
                    if use_scale_shift_norm
                    else self.out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward,
            (x, emb),
            self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward,
            (x,),
            self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


@dataclass(eq=False)
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    in_channels: int
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (1, 2, 2, 2)
    dropout: float = 0.0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    with_value_fourier_features: bool = False
    with_coordinate_fourier_features: bool = False
    fourier_feature_start: int = 6
    fourier_feature_stop: int = 8
    fourier_feature_step: int = 1
    coordinate_fourier_start: int = 0
    coordinate_fourier_stop: int = 4
    coordinate_fourier_step: int = 1
    coordinate_fourier_include_raw_coords: bool = True
    coordinate_fourier_coord_range: str = "unit"
    ignore_time: bool = False
    input_projection: bool = True
    scalar_conditioning: bool = False
    scalar_conditioning_dim: int = 0

    image_size: int = -1  # not used...
    _target_: str = "lib.models.gd_unet.UNetModel"

    def __post_init__(self):
        super().__init__()

        self.data_in_channels = int(self.in_channels)
        self.with_value_fourier_features = bool(self.with_value_fourier_features)
        self.with_coordinate_fourier_features = bool(self.with_coordinate_fourier_features)
        self.scalar_conditioning = bool(self.scalar_conditioning)
        self.scalar_conditioning_dim = int(self.scalar_conditioning_dim or 0)

        self.value_fourier_feature_channels = 0
        if self.with_value_fourier_features:
            self.value_fourier_feature_channels = fourier_feature_channel_count(
                self.data_in_channels,
                start=self.fourier_feature_start,
                stop=self.fourier_feature_stop,
                step=self.fourier_feature_step,
            )
        self.coordinate_fourier_feature_channels = 0
        if self.with_coordinate_fourier_features:
            self.coordinate_fourier_feature_channels = coordinate_fourier_feature_channel_count(
                spatial_dims=2,
                start=self.coordinate_fourier_start,
                stop=self.coordinate_fourier_stop,
                step=self.coordinate_fourier_step,
                include_raw_coords=self.coordinate_fourier_include_raw_coords,
            )
        self.fourier_feature_channels = (
            self.value_fourier_feature_channels + self.coordinate_fourier_feature_channels
        )
        self.in_channels = self.data_in_channels + self.fourier_feature_channels

        self.time_embed_dim = self.model_channels * 4
        if self.ignore_time:
            self.time_embed = lambda x: torch.zeros(
                x.shape[0], self.time_embed_dim, device=x.device, dtype=x.dtype
            )
        else:
            self.time_embed = nn.Sequential(
                linear(self.model_channels, self.time_embed_dim),
                nn.SiLU(),
                linear(self.time_embed_dim, self.time_embed_dim),
            )

        self.scalar_embed = None
        if self.scalar_conditioning:
            if self.scalar_conditioning_dim <= 0:
                raise ValueError(
                    "scalar_conditioning=True requires scalar_conditioning_dim > 0"
                )
            self.scalar_embed = nn.Sequential(
                linear(self.scalar_conditioning_dim, self.time_embed_dim),
                nn.SiLU(),
                linear(self.time_embed_dim, self.time_embed_dim),
            )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(
                self.num_classes + 1, self.time_embed_dim, padding_idx=self.num_classes
            )

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        if self.input_projection:
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(self.dims, self.in_channels, ch, 3, padding=1)
                    )
                ]
            )
        else:
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(torch.nn.Identity())]
            )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                        )
                        if self.resblock_updown
                        else Upsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, extra=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if extra is None:
            extra = {}

        if "concat_conditioning" in extra:
            raise ValueError(
                "concat_conditioning requires explicit concat_conditioning_channels in the model config; "
                "implicit channel expansion is not supported."
            )

        original_x = x
        features = [x]
        if self.with_value_fourier_features:
            features.append(
                base2_fourier_features(
                    original_x,
                    start=self.fourier_feature_start,
                    stop=self.fourier_feature_stop,
                    step=self.fourier_feature_step,
                )
            )
        if self.with_coordinate_fourier_features:
            features.append(
                coordinate_fourier_features(
                    original_x,
                    start=self.coordinate_fourier_start,
                    stop=self.coordinate_fourier_stop,
                    step=self.coordinate_fourier_step,
                    include_raw_coords=self.coordinate_fourier_include_raw_coords,
                    coord_range=self.coordinate_fourier_coord_range,
                )
            )
        if len(features) > 1:
            x = torch.cat(features, dim=1)

        hs = []
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x))

        if self.ignore_time:
            emb = emb * 0.0

        if self.scalar_conditioning:
            if "scalar_conditioning" not in extra:
                raise ValueError(
                    "UNetModel scalar_conditioning=True requires "
                    "extra['scalar_conditioning'] with shape [B,K]"
                )
            scalar = extra["scalar_conditioning"]
            if not torch.is_tensor(scalar):
                scalar = torch.as_tensor(scalar, device=x.device)
            scalar = scalar.to(device=x.device, dtype=emb.dtype)
            if scalar.ndim != 2:
                raise ValueError(
                    "extra['scalar_conditioning'] must have shape [B,K], "
                    f"got {tuple(scalar.shape)}"
                )
            if int(scalar.shape[0]) != int(x.shape[0]):
                raise ValueError(
                    "extra['scalar_conditioning'] batch size must match input; "
                    f"got {int(scalar.shape[0])} vs {int(x.shape[0])}"
                )
            if int(scalar.shape[1]) != int(self.scalar_conditioning_dim):
                raise ValueError(
                    "extra['scalar_conditioning'] feature dimension must match "
                    f"scalar_conditioning_dim={self.scalar_conditioning_dim}; "
                    f"got {int(scalar.shape[1])}"
                )
            emb = emb + self.scalar_embed(scalar)

        if self.num_classes and "label" not in extra:
            # Hack to deal with ddp find_unused_parameters not working with activation checkpointing...
            # self.num_classes corresponds to the pad index of the embedding table
            extra["label"] = torch.full(
                (x.size(0),), self.num_classes, dtype=torch.long, device=x.device
            )

        if self.num_classes is not None and "label" in extra:
            y = extra["label"]
            assert (
                y.shape == x.shape[:1]
            ), f"Labels have shape {y.shape}, which does not match the batch dimension of the input {x.shape}"
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        result = self.out(h)
        return result


# Based on https://github.com/google-research/vdm/blob/main/model_vdm.py
def fourier_feature_channel_count(
    in_channels: int,
    start: int = 0,
    stop: int = 8,
    step: int = 1,
) -> int:
    if step == 0:
        raise ValueError("fourier feature step must be non-zero")
    return int(in_channels) * len(range(int(start), int(stop), int(step))) * 2


def base2_fourier_features(
    inputs: torch.Tensor, start: int = 0, stop: int = 8, step: int = 1
) -> torch.Tensor:
    freqs = torch.arange(start, stop, step, device=inputs.device, dtype=inputs.dtype)

    # Create Base 2 Fourier features
    w = 2.0**freqs * 2 * np.pi
    w = torch.tile(w[None, :], (1, inputs.size(1)))

    # Compute features
    h = torch.repeat_interleave(inputs, len(freqs), dim=1)
    h = w[:, :, None, None] * h
    h = torch.cat([torch.sin(h), torch.cos(h)], dim=1)
    return h


def coordinate_fourier_feature_channel_count(
    spatial_dims: int = 2,
    start: int = 0,
    stop: int = 4,
    step: int = 1,
    include_raw_coords: bool = True,
) -> int:
    if step == 0:
        raise ValueError("coordinate fourier feature step must be non-zero")
    num_freqs = len(range(int(start), int(stop), int(step)))
    raw_channels = int(spatial_dims) if include_raw_coords else 0
    return raw_channels + int(spatial_dims) * num_freqs * 2


def coordinate_fourier_features(
    inputs: torch.Tensor,
    start: int = 0,
    stop: int = 4,
    step: int = 1,
    include_raw_coords: bool = True,
    coord_range: str = "unit",
) -> torch.Tensor:
    if inputs.dim() != 4:
        raise ValueError(
            f"coordinate_fourier_features expects BCHW inputs, got shape {tuple(inputs.shape)}"
        )
    if step == 0:
        raise ValueError("coordinate fourier feature step must be non-zero")
    if coord_range == "unit":
        coord_min, coord_max = 0.0, 1.0
    elif coord_range == "minus_one_one":
        coord_min, coord_max = -1.0, 1.0
    else:
        raise ValueError("coordinate_fourier_coord_range must be 'unit' or 'minus_one_one'")

    batch, _, height, width = inputs.shape
    y = torch.linspace(coord_min, coord_max, height, device=inputs.device, dtype=inputs.dtype)
    x = torch.linspace(coord_min, coord_max, width, device=inputs.device, dtype=inputs.dtype)
    yy = y[:, None].expand(height, width)
    xx = x[None, :].expand(height, width)
    coords = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    features = []
    if include_raw_coords:
        features.append(coords)

    freqs = torch.arange(start, stop, step, device=inputs.device, dtype=inputs.dtype)
    if freqs.numel() > 0:
        omega = (2.0**freqs) * 2 * np.pi
        angles = coords.unsqueeze(2) * omega.view(1, 1, -1, 1, 1)
        sin_cos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=2)
        features.append(sin_cos.reshape(batch, -1, height, width))

    if not features:
        return inputs.new_empty((batch, 0, height, width))
    return torch.cat(features, dim=1)
