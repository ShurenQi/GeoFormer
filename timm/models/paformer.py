from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ._geoformer_common import (
    ATTENTION_HYBRID_SCALES,
    attention_hybrid_kwargs,
    build_geoformer_family_model,
    paper_default_cfgs,
)
from ._registry import register_model
from .metaformer import MetaFormer, StarReLU


class PoolMixer(nn.Module):
    def __init__(self, dim: int, pool_size: int = 3, **_: object):
        super().__init__()
        if pool_size % 2 != 1:
            raise ValueError(f"PoolMixer requires an odd pool size; got {pool_size}.")
        self.channels = int(dim)
        self.pool_size = int(pool_size)
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.pool(features)


class PoolBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        inverted_bottleneck_ratio: float = 2.0,
        activation_layer=StarReLU,
        pool_size: int = 3,
        **_: object,
    ):
        super().__init__()
        intermediate_channels = int(inverted_bottleneck_ratio * dim)
        if intermediate_channels <= 0:
            raise ValueError(
                "PoolBlock inverted bottleneck produced "
                f"{intermediate_channels} channels for dim={dim}."
            )
        self.input_projection = nn.Conv2d(
            dim, intermediate_channels, kernel_size=1, bias=False
        )
        self.input_activation = activation_layer()
        self.spatial_mixer = PoolMixer(intermediate_channels, pool_size=pool_size)
        self.output_projection = nn.Conv2d(
            intermediate_channels, dim, kernel_size=1, bias=False
        )

    def forward(self, features: Tensor) -> Tensor:
        features = self.input_projection(features)
        features = self.input_activation(features)
        features = self.spatial_mixer(features)
        return self.output_projection(features)


class PAFormer(MetaFormer):
    pass


_MODEL_NAMES = tuple(f"paformer_{scale}" for scale in ATTENTION_HYBRID_SCALES)
default_cfgs = paper_default_cfgs(_MODEL_NAMES, mlp_head=True)


def _create_paformer(scale_name: str, pretrained: bool, **kwargs) -> PAFormer:
    scale = ATTENTION_HYBRID_SCALES[scale_name]
    model_name = f"paformer_{scale_name}"
    model_kwargs = attention_hybrid_kwargs(scale, PoolBlock, **kwargs)
    return build_geoformer_family_model(
        PAFormer,
        model_name,
        pretrained,
        **model_kwargs,
    )


@register_model
def paformer_s18(pretrained: bool = False, **kwargs) -> PAFormer:
    return _create_paformer("s18", pretrained, **kwargs)


@register_model
def paformer_s36(pretrained: bool = False, **kwargs) -> PAFormer:
    return _create_paformer("s36", pretrained, **kwargs)


@register_model
def paformer_m36(pretrained: bool = False, **kwargs) -> PAFormer:
    return _create_paformer("m36", pretrained, **kwargs)


__all__ = ["PAFormer", "PoolBlock", "PoolMixer", *_MODEL_NAMES]
