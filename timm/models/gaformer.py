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
from .geoformer import GeoMixer, W2_DETAIL_INDICES
from .metaformer import MetaFormer, StarReLU


class GeoBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        inverted_bottleneck_ratio: float = 2.0,
        activation_layer=StarReLU,
        **_: object,
    ):
        super().__init__()
        intermediate_channels = int(inverted_bottleneck_ratio * dim)
        if intermediate_channels <= 0:
            raise ValueError(
                "GeoBlock inverted bottleneck produced "
                f"{intermediate_channels} channels for dim={dim}."
            )
        self.input_projection = nn.Conv2d(
            dim, intermediate_channels, kernel_size=1, bias=False
        )
        self.input_activation = activation_layer()
        self.spatial_mixer = GeoMixer(
            dim=intermediate_channels,
            proj_drop=0.0,
            kernel_size=5,
            branch_indices=W2_DETAIL_INDICES,
            residual_form=False,
        )
        self.output_projection = nn.Conv2d(
            intermediate_channels, dim, kernel_size=1, bias=False
        )

    def forward(self, features: Tensor) -> Tensor:
        features = self.input_projection(features)
        features = self.input_activation(features)
        features = self.spatial_mixer(features)
        return self.output_projection(features)


class GAFormer(MetaFormer):
    pass


_GA_NAMES = tuple(f"gaformer_{scale}" for scale in ATTENTION_HYBRID_SCALES)
default_cfgs = paper_default_cfgs(_GA_NAMES, mlp_head=True)


def _create_gaformer(
    scale_name: str,
    pretrained: bool,
    **kwargs,
) -> GAFormer:
    scale = ATTENTION_HYBRID_SCALES[scale_name]
    model_name = f"gaformer_{scale_name}"
    model_kwargs = attention_hybrid_kwargs(scale, GeoBlock, **kwargs)
    return build_geoformer_family_model(
        GAFormer,
        model_name,
        pretrained,
        **model_kwargs,
    )


@register_model
def gaformer_s18(pretrained: bool = False, **kwargs) -> GAFormer:
    return _create_gaformer("s18", pretrained, **kwargs)


@register_model
def gaformer_s36(pretrained: bool = False, **kwargs) -> GAFormer:
    return _create_gaformer("s36", pretrained, **kwargs)


@register_model
def gaformer_m36(pretrained: bool = False, **kwargs) -> GAFormer:
    return _create_gaformer("m36", pretrained, **kwargs)


__all__ = ["GAFormer", "GeoBlock", *_GA_NAMES]
