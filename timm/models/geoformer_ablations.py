from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ._registry import register_model

from ._geoformer_common import (
    GEOFORMER_SCALES,
    build_geoformer_family_model,
    geoformer_kwargs,
    paper_default_cfgs,
)
from .geoformer import GeoFormer, GeoMixer, default_branch_indices


class PointwiseFusion(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels, output_channels, kernel_size=1, bias=False
        )
        self.normalization = nn.BatchNorm2d(output_channels)
        self.activation = nn.GELU()

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(self.normalization(self.projection(features)))


class MultiScaleGeoMixer(nn.Module):
    def __init__(self, dim: int, proj_drop: float = 0.0, **kwargs):
        super().__init__()
        kwargs.pop("use_bn", None)
        self.scale_mixers = nn.ModuleList(
            [
                GeoMixer(
                    dim,
                    proj_drop=0.0,
                    kernel_size=kernel_size,
                    branch_indices=default_branch_indices(kernel_size),
                    maximum_angular_order=2,
                    **kwargs,
                )
                for kernel_size in (5, 7)
            ]
        )
        self.scale_fusion = PointwiseFusion(dim * 2, dim)
        self.projection_dropout = (
            nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        )

    def forward(self, features: Tensor) -> Tensor:
        scale_responses = [mixer(features) for mixer in self.scale_mixers]
        return self.projection_dropout(
            self.scale_fusion(torch.cat(scale_responses, dim=1))
        )


class HigherOrderGeoMixer(GeoMixer):
    def __init__(self, dim: int, proj_drop: float = 0.0, **kwargs):
        kwargs.pop("use_bn", None)
        super().__init__(
            dim,
            proj_drop=proj_drop,
            kernel_size=7,
            branch_indices=default_branch_indices(7, higher_order=True),
            maximum_angular_order=3,
            **kwargs,
        )
        self.detail_fusion = PointwiseFusion(dim, dim)

    def detail_response(self, features: Tensor) -> Tensor:
        return self.detail_fusion(super().detail_response(features))


_MODEL_NAMES = (
    "geoformer_s12_multiscale",
    "geoformer_m48_multiscale",
    "geoformer_s12_higher_order",
    "geoformer_m48_higher_order",
)
default_cfgs = paper_default_cfgs(_MODEL_NAMES, crop_pct=0.875)


def _create_ablation(scale_name: str, mixer, suffix: str, pretrained: bool, **kwargs):
    scale = GEOFORMER_SCALES[scale_name]
    model_name = f"geoformer_{scale_name}_{suffix}"
    return build_geoformer_family_model(
        GeoFormer,
        model_name,
        pretrained,
        **geoformer_kwargs(scale, mixer, **kwargs),
    )


@register_model
def geoformer_s12_multiscale(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_ablation(
        "s12", MultiScaleGeoMixer, "multiscale", pretrained, **kwargs
    )


@register_model
def geoformer_m48_multiscale(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_ablation(
        "m48", MultiScaleGeoMixer, "multiscale", pretrained, **kwargs
    )


@register_model
def geoformer_s12_higher_order(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_ablation(
        "s12", HigherOrderGeoMixer, "higher_order", pretrained, **kwargs
    )


@register_model
def geoformer_m48_higher_order(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_ablation(
        "m48", HigherOrderGeoMixer, "higher_order", pretrained, **kwargs
    )


__all__ = [
    "HigherOrderGeoMixer",
    "MultiScaleGeoMixer",
    "PointwiseFusion",
    *_MODEL_NAMES,
]
