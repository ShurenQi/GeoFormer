from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._geoformer_common import (
    GEOFORMER_SCALES,
    build_geoformer_family_model,
    geoformer_kwargs,
    paper_default_cfgs,
)
from ._registry import register_model
from .metaformer import MetaFormer


@dataclass(frozen=True, order=True)
class GeoBranchIndex:
    radial_order: int
    angular_order: int


W2_DETAIL_INDICES: tuple[GeoBranchIndex, ...] = tuple(
    GeoBranchIndex(radial_order, angular_order)
    for radial_order, angular_order in (
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    )
)


def default_branch_indices(
    kernel_size: int,
    *,
    higher_order: bool = False,
) -> tuple[GeoBranchIndex, ...]:
    if kernel_size == 3:
        pairs = ((1, 0), (0, 1), (1, 1))
    elif kernel_size == 5:
        return W2_DETAIL_INDICES
    elif kernel_size == 7:
        pairs = (
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
        )
        if higher_order:
            pairs += ((1, 3), (2, 3), (3, 3))
    else:
        raise ValueError(
            f"GeoMixer supports odd kernel sizes 3, 5, and 7; got {kernel_size}."
        )
    return tuple(GeoBranchIndex(*pair) for pair in pairs)


def build_geo_basis_kernel(
    kernel_size: int,
    radial_order: int,
    angular_order: int,
    *,
    angular_component: str = "cos",
) -> Tensor:
    if kernel_size % 2 != 1:
        raise ValueError(f"GeoBasisKernel requires an odd size; got {kernel_size}.")
    if angular_component not in {"cos", "sin"}:
        raise ValueError("angular_component must be 'cos' or 'sin'.")

    support_radius = (kernel_size - 1) / 2.0
    coordinates = torch.arange(
        -(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32
    )
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    angle = torch.atan2(yy, xx)
    normalized_radius = radius / support_radius

    radial_response = torch.cos(math.pi * radial_order * normalized_radius)
    radial_response = radial_response * (normalized_radius <= 1.0 + 1e-6)
    angular_response = (
        torch.cos(angular_order * angle)
        if angular_component == "cos"
        else torch.sin(angular_order * angle)
    )
    kernel = radial_response * angular_response
    l1_norm = kernel.abs().sum()
    return torch.where(l1_norm > 1e-6, kernel / l1_norm, kernel)


class GeoBasisKernel(nn.Module):
    def __init__(self, channels: int, kernel: Tensor):
        super().__init__()
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("GeoBasisKernel expects a square 2-D kernel.")
        kernel_size = int(kernel.shape[0])
        if kernel_size % 2 != 1:
            raise ValueError("GeoBasisKernel expects an odd kernel size.")
        weight = kernel.float().reshape(1, 1, kernel_size, kernel_size)
        self.register_buffer(
            "weight",
            weight.repeat(int(channels), 1, 1, 1),
            persistent=True,
        )
        self.channels = int(channels)
        self.padding = kernel_size // 2

    def forward(self, features: Tensor) -> Tensor:
        return F.conv2d(
            features,
            self.weight,
            padding=self.padding,
            groups=self.channels,
        )


class GeoMagnitudeBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        branch_index: GeoBranchIndex,
        *,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.branch_index = branch_index
        self.epsilon = float(epsilon)
        if branch_index.angular_order == 0:
            kernel = build_geo_basis_kernel(
                kernel_size,
                branch_index.radial_order,
                0,
            )
            self.scalar_filter = GeoBasisKernel(channels, kernel)
        else:
            cosine = build_geo_basis_kernel(
                kernel_size,
                branch_index.radial_order,
                branch_index.angular_order,
                angular_component="cos",
            )
            sine = build_geo_basis_kernel(
                kernel_size,
                branch_index.radial_order,
                branch_index.angular_order,
                angular_component="sin",
            )
            self.cosine_filter = GeoBasisKernel(channels, cosine)
            self.sine_filter = GeoBasisKernel(channels, sine)

    def forward(self, features: Tensor) -> Tensor:
        if self.branch_index.angular_order == 0:
            return self.scalar_filter(features).abs()
        cosine_response = self.cosine_filter(features)
        sine_response = self.sine_filter(features)
        return torch.sqrt(
            cosine_response.square() + sine_response.square() + self.epsilon
        )


class GeoMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        *,
        kernel_size: int = 5,
        branch_indices: Sequence[GeoBranchIndex | tuple[int, int]] | None = None,
        minimum_channels_per_branch: int = 8,
        detail_scale_init: float = 3e-3,
        epsilon: float = 1e-4,
        maximum_angular_order: int = 2,
        residual_form: bool = True,
        normalize_details: bool = False,
        **_: object,
    ):
        super().__init__()
        if kernel_size not in {3, 5, 7}:
            raise ValueError(f"Unsupported GeoMixer kernel size: {kernel_size}.")

        self.channels = int(dim)
        self.kernel_size = int(kernel_size)
        self.residual_form = bool(residual_form)
        requested_indices = branch_indices or default_branch_indices(kernel_size)
        requested_indices = tuple(
            index if isinstance(index, GeoBranchIndex) else GeoBranchIndex(*index)
            for index in requested_indices
        )

        zero_kernel = build_geo_basis_kernel(kernel_size, 0, 0)
        self.zeroth_order_filter = GeoBasisKernel(self.channels, zero_kernel)

        maximum_radial_order = kernel_size // 2
        eligible_indices = sorted(
            (
                index
                for index in requested_indices
                if index != GeoBranchIndex(0, 0)
                and index.radial_order <= maximum_radial_order
                and index.angular_order <= maximum_angular_order
            ),
            key=lambda index: (index.angular_order, index.radial_order),
        )
        maximum_branches = max(
            1, self.channels // max(1, int(minimum_channels_per_branch))
        )
        self.branch_indices = tuple(eligible_indices[:maximum_branches])
        if not self.branch_indices:
            raise ValueError("GeoMixer requires at least one non-zero branch.")

        base_width, remainder = divmod(self.channels, len(self.branch_indices))
        self.channel_splits = tuple(
            base_width + int(index < remainder)
            for index in range(len(self.branch_indices))
        )
        self.magnitude_branches = nn.ModuleList(
            GeoMagnitudeBranch(
                channels=branch_width,
                kernel_size=kernel_size,
                branch_index=branch_index,
                epsilon=epsilon,
            )
            for branch_width, branch_index in zip(
                self.channel_splits,
                self.branch_indices,
            )
        )
        self.branch_norm = (
            nn.BatchNorm2d(self.channels) if normalize_details else nn.Identity()
        )
        self.detail_scale = nn.Parameter(
            torch.full((1, self.channels, 1, 1), float(detail_scale_init))
        )
        self.projection_dropout = (
            nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        )

    def zeroth_order_response(self, features: Tensor) -> Tensor:
        response = self.zeroth_order_filter(features)
        return response - features if self.residual_form else response

    def detail_response(self, features: Tensor) -> Tensor:
        channel_groups = torch.split(features, self.channel_splits, dim=1)
        branch_responses = [
            branch(group)
            for branch, group in zip(self.magnitude_branches, channel_groups)
        ]
        return self.branch_norm(torch.cat(branch_responses, dim=1))

    def forward(self, features: Tensor) -> Tensor:
        mixed = self.zeroth_order_response(features)
        mixed = mixed + self.detail_scale * self.detail_response(features)
        return self.projection_dropout(mixed)


class GeoFormer(MetaFormer):
    pass


_MODEL_NAMES = tuple(f"geoformer_{scale}" for scale in GEOFORMER_SCALES)
default_cfgs = paper_default_cfgs(_MODEL_NAMES, crop_pct=0.875)


def _create_geoformer(scale_name: str, pretrained: bool, **kwargs) -> GeoFormer:
    scale = GEOFORMER_SCALES[scale_name]
    model_name = f"geoformer_{scale_name}"
    model_kwargs = geoformer_kwargs(scale, GeoMixer, **kwargs)
    return build_geoformer_family_model(
        GeoFormer,
        model_name,
        pretrained,
        **model_kwargs,
    )


@register_model
def geoformer_s12(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_geoformer("s12", pretrained, **kwargs)


@register_model
def geoformer_s24(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_geoformer("s24", pretrained, **kwargs)


@register_model
def geoformer_s36(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_geoformer("s36", pretrained, **kwargs)


@register_model
def geoformer_m36(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_geoformer("m36", pretrained, **kwargs)


@register_model
def geoformer_m48(pretrained: bool = False, **kwargs) -> GeoFormer:
    return _create_geoformer("m48", pretrained, **kwargs)


__all__ = [
    "GeoBasisKernel",
    "GeoMagnitudeBranch",
    "GeoMixer",
    "GeoBranchIndex",
    "GeoFormer",
    "W2_DETAIL_INDICES",
    "build_geo_basis_kernel",
    "default_branch_indices",
    *_MODEL_NAMES,
]
