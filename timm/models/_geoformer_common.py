from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Type

import torch.nn as nn

from ..layers import GroupNorm1
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs
from .metaformer import (
    Attention,
    LayerNorm2dNoBias,
    LayerNormNoBias,
    MetaFormer,
    _cfg,
    checkpoint_filter_fn,
)


@dataclass(frozen=True)
class ScaleSpec:
    channels: tuple[int, int, int, int]
    blocks_per_stage: tuple[int, int, int, int]
    layer_scale_init: float | None = None


GEOFORMER_SCALES = {
    "s12": ScaleSpec((64, 128, 320, 512), (2, 2, 6, 2), 1e-5),
    "s24": ScaleSpec((64, 128, 320, 512), (4, 4, 12, 4), 1e-5),
    "s36": ScaleSpec((64, 128, 320, 512), (6, 6, 18, 6), 1e-6),
    "m36": ScaleSpec((96, 192, 384, 768), (6, 6, 18, 6), 1e-6),
    "m48": ScaleSpec((96, 192, 384, 768), (8, 8, 24, 8), 1e-6),
}


ATTENTION_HYBRID_SCALES = {
    "s18": ScaleSpec((64, 128, 320, 512), (3, 3, 9, 3)),
    "s36": ScaleSpec((64, 128, 320, 512), (3, 12, 18, 3)),
    "m36": ScaleSpec((96, 192, 384, 576), (3, 12, 18, 3)),
}


def paper_default_cfgs(
    model_names: Iterable[str],
    *,
    mlp_head: bool = False,
    crop_pct: float = 1.0,
):
    classifier = "head.fc.fc2" if mlp_head else "head.fc"
    return generate_default_cfgs(
        {
            f"{name}.in1k": _cfg(classifier=classifier, crop_pct=crop_pct)
            for name in model_names
        }
    )


def build_geoformer_family_model(
    model_class: Type[MetaFormer],
    model_name: str,
    pretrained: bool,
    **model_kwargs,
) -> MetaFormer:
    default_indices = tuple(range(len(model_kwargs.get("depths", (2, 2, 6, 2)))))
    out_indices = model_kwargs.pop("out_indices", default_indices)
    return build_model_with_cfg(
        model_class,
        model_name,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg={"flatten_sequential": True, "out_indices": out_indices},
        **model_kwargs,
    )


def geoformer_kwargs(
    scale: ScaleSpec,
    geo_mixer,
    **overrides,
) -> dict:
    return {
        "depths": list(scale.blocks_per_stage),
        "dims": list(scale.channels),
        "token_mixers": geo_mixer,
        "mlp_act": nn.GELU,
        "mlp_bias": True,
        "norm_layers": GroupNorm1,
        "layer_scale_init_values": scale.layer_scale_init,
        "res_scale_init_values": None,
        "use_mlp_head": False,
        **overrides,
    }


def attention_hybrid_kwargs(
    scale: ScaleSpec,
    early_stage_mixer,
    **overrides,
) -> dict:
    return {
        "depths": list(scale.blocks_per_stage),
        "dims": list(scale.channels),
        "token_mixers": [early_stage_mixer, early_stage_mixer, Attention, Attention],
        "norm_layers": [LayerNorm2dNoBias] * 2 + [LayerNormNoBias] * 2,
        **overrides,
    }


__all__ = [
    "ATTENTION_HYBRID_SCALES",
    "GEOFORMER_SCALES",
    "ScaleSpec",
    "attention_hybrid_kwargs",
    "build_geoformer_family_model",
    "geoformer_kwargs",
    "paper_default_cfgs",
]
