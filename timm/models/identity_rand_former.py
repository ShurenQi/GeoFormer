from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..layers import GroupNorm1
from ._registry import register_model
from .metaformer import MetaFormer

from ._geoformer_common import (
    GEOFORMER_SCALES,
    build_geoformer_family_model,
    paper_default_cfgs,
)


class IdentityMixer(nn.Module):
    def __init__(self, dim: int | None = None, **_: object):
        super().__init__()
        self.channels = None if dim is None else int(dim)

    def forward(self, features: Tensor) -> Tensor:
        return features


class FixedRandomMixer(nn.Module):
    token_count: int = -1

    def __init__(self, dim: int, seed: int = 0, proj_drop: float = 0.0, **_: object):
        super().__init__()
        if self.token_count <= 0:
            raise ValueError("FixedRandomMixer subclasses must define token_count.")
        generator = torch.Generator().manual_seed(int(seed))
        matrix = torch.rand(self.token_count, self.token_count, generator=generator)
        matrix = torch.softmax(matrix, dim=-1)
        self.mixing_matrix = nn.Parameter(matrix, requires_grad=False)
        self.projection_dropout = (
            nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(
                f"FixedRandomMixer expects NCHW input, got {features.shape}."
            )
        batch, channels, height, width = features.shape
        if height * width != self.token_count:
            raise ValueError(
                f"Expected {self.token_count} tokens, got {height}x{width}. "
                "RandFormer uses the paper's fixed 224x224 input resolution."
            )
        tokens = features.reshape(batch, channels, self.token_count).transpose(1, 2)
        matrix = self.mixing_matrix.to(device=tokens.device, dtype=tokens.dtype)
        tokens = torch.einsum("mn,bnc->bmc", matrix, tokens)
        tokens = self.projection_dropout(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class Stage3RandomMixer(FixedRandomMixer):
    token_count = 14 * 14


class Stage4RandomMixer(FixedRandomMixer):
    token_count = 7 * 7


class IdentityFormer(MetaFormer):
    pass


class RandFormer(MetaFormer):
    pass


_SCALES = tuple(GEOFORMER_SCALES)
_IDENTITY_NAMES = tuple(f"identityformer_{scale}" for scale in _SCALES)
_RAND_NAMES = tuple(f"randformer_{scale}" for scale in _SCALES)
default_cfgs = paper_default_cfgs((*_IDENTITY_NAMES, *_RAND_NAMES), crop_pct=0.875)


def _baseline_kwargs(scale_name: str, token_mixers, **overrides) -> dict:
    scale = GEOFORMER_SCALES[scale_name]
    return {
        "depths": list(scale.blocks_per_stage),
        "dims": list(scale.channels),
        "token_mixers": token_mixers,
        "downsample_norm": None,
        "mlp_act": nn.GELU,
        "mlp_bias": True,
        "norm_layers": GroupNorm1,
        "layer_scale_init_values": 1e-6,
        "res_scale_init_values": None,
        "use_mlp_head": False,
        **overrides,
    }


def _create_identityformer(scale_name: str, pretrained: bool, **kwargs):
    model_name = f"identityformer_{scale_name}"
    return build_geoformer_family_model(
        IdentityFormer,
        model_name,
        pretrained,
        **_baseline_kwargs(scale_name, IdentityMixer, **kwargs),
    )


def _create_randformer(scale_name: str, pretrained: bool, **kwargs):
    model_name = f"randformer_{scale_name}"
    mixers = [IdentityMixer, IdentityMixer, Stage3RandomMixer, Stage4RandomMixer]
    return build_geoformer_family_model(
        RandFormer,
        model_name,
        pretrained,
        **_baseline_kwargs(scale_name, mixers, **kwargs),
    )


@register_model
def identityformer_s12(pretrained: bool = False, **kwargs) -> IdentityFormer:
    return _create_identityformer("s12", pretrained, **kwargs)


@register_model
def identityformer_s24(pretrained: bool = False, **kwargs) -> IdentityFormer:
    return _create_identityformer("s24", pretrained, **kwargs)


@register_model
def identityformer_s36(pretrained: bool = False, **kwargs) -> IdentityFormer:
    return _create_identityformer("s36", pretrained, **kwargs)


@register_model
def identityformer_m36(pretrained: bool = False, **kwargs) -> IdentityFormer:
    return _create_identityformer("m36", pretrained, **kwargs)


@register_model
def identityformer_m48(pretrained: bool = False, **kwargs) -> IdentityFormer:
    return _create_identityformer("m48", pretrained, **kwargs)


@register_model
def randformer_s12(pretrained: bool = False, **kwargs) -> RandFormer:
    return _create_randformer("s12", pretrained, **kwargs)


@register_model
def randformer_s24(pretrained: bool = False, **kwargs) -> RandFormer:
    return _create_randformer("s24", pretrained, **kwargs)


@register_model
def randformer_s36(pretrained: bool = False, **kwargs) -> RandFormer:
    return _create_randformer("s36", pretrained, **kwargs)


@register_model
def randformer_m36(pretrained: bool = False, **kwargs) -> RandFormer:
    return _create_randformer("m36", pretrained, **kwargs)


@register_model
def randformer_m48(pretrained: bool = False, **kwargs) -> RandFormer:
    return _create_randformer("m48", pretrained, **kwargs)


__all__ = [
    "FixedRandomMixer",
    "IdentityFormer",
    "IdentityMixer",
    "RandFormer",
    "Stage3RandomMixer",
    "Stage4RandomMixer",
    *_IDENTITY_NAMES,
    *_RAND_NAMES,
]
