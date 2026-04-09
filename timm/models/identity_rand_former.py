# timm/models/identity_rand_former.py
"""
IdentityFormer / RandFormer for timm MetaFormer (no modifications to metaformer.py).

Paper-aligned definitions (arXiv:2210.13452):
- IdentityFormer: T = (Id, Id, Id, Id)
- RandFormer:     T = (Id, Id, Rand, Rand), with global random mixing WR X

This version **forces 224x224** inputs by using fixed token-length random matrices:
- Stage 3 token length: 14*14 = 196
- Stage 4 token length:  7*7 =  49
If a different resolution is used, Rand mixing will raise an AssertionError.

IMPORTANT: This file must be imported at runtime for @register_model to execute.
Example:
  import timm
  import timm.models.identity_rand_former
  m = timm.create_model('randformer_m48', pretrained=False, seed=0)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from timm.layers import GroupNorm1
from ._registry import register_model
from .metaformer import MetaFormer, _create_metaformer

__all__ = [
    "IdentityMixing",
    "RandomMixing196",
    "RandomMixing49",
    "identityformer_s12",
    "identityformer_s24",
    "identityformer_s36",
    "identityformer_m36",
    "identityformer_m48",
    "randformer_s12",
    "randformer_s24",
    "randformer_s36",
    "randformer_m36",
    "randformer_m48",
]


class IdentityMixing(nn.Module):
    """Identity mapping token mixer: f(X)=X (no token mixing)."""

    def __init__(self, dim=None, proj_drop: float = 0.0, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FixedTokenRandomMixing(nn.Module):
    """Global random mixing with frozen W_R in R^{N x N}, fixed N (=H*W).

    Implements Algorithm-style:
      W_R = softmax(rand(N,N), dim=-1), frozen
      out = W_R @ X

    Enforces fixed token length at runtime (thereby forcing 224 input for MetaFormer).
    """
    NUM_TOKENS: int = -1  # override in subclasses

    def __init__(self, dim: int, seed: int = 0, proj_drop: float = 0.0, **kwargs):
        super().__init__()
        if self.NUM_TOKENS <= 0:
            raise ValueError("NUM_TOKENS must be set in subclass.")

        g = torch.Generator()
        g.manual_seed(int(seed))

        wr = torch.softmax(torch.rand(self.NUM_TOKENS, self.NUM_TOKENS, generator=g), dim=-1)
        self.random_matrix = nn.Parameter(wr, requires_grad=False)
        self.drop = nn.Dropout(proj_drop) if proj_drop and proj_drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm MetaFormer non-attention mixers use NCHW
        if x.dim() != 4:
            raise ValueError(f"Expected NCHW 4D tensor, got shape={tuple(x.shape)}")

        b, c, h, w = x.shape
        n = h * w
        if n != self.NUM_TOKENS:
            raise AssertionError(
                f"RandFormer (fixed-224) expects N=H*W={self.NUM_TOKENS} tokens at this stage, "
                f"but got H={h}, W={w}, N={n}. Ensure input is 224x224."
            )

        tokens = x.reshape(b, c, n).transpose(1, 2)  # [B, N, C]
        wr = self.random_matrix.to(dtype=tokens.dtype, device=tokens.device)
        tokens = torch.einsum("mn,bnc->bmc", wr, tokens)
        tokens = self.drop(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class RandomMixing196(_FixedTokenRandomMixing):
    """Fixed random mixing for 14x14 tokens (stage 3 at 224 input)."""
    NUM_TOKENS = 14 * 14


class RandomMixing49(_FixedTokenRandomMixing):
    """Fixed random mixing for 7x7 tokens (stage 4 at 224 input)."""
    NUM_TOKENS = 7 * 7


def _recipe_kwargs(depths, dims, token_mixers, **kwargs):
    """Match PoolFormer v1 recipe used in metaformer.py for fair comparison."""
    return dict(
        depths=depths,
        dims=dims,
        token_mixers=token_mixers,
        downsample_norm=None,
        mlp_act=nn.GELU,
        mlp_bias=True,
        norm_layers=GroupNorm1,
        layer_scale_init_values=1e-6,
        res_scale_init_values=None,
        use_mlp_head=False,
        **kwargs,
    )


# -------------------------
# IdentityFormer (T = Id,Id,Id,Id)
# -------------------------

@register_model
def identityformer_s12(pretrained: bool = False, **kwargs) -> MetaFormer:
    mk = _recipe_kwargs([2, 2, 6, 2], [64, 128, 320, 512], IdentityMixing, **kwargs)
    return _create_metaformer("identityformer_s12", pretrained=pretrained, **mk)


@register_model
def identityformer_s24(pretrained: bool = False, **kwargs) -> MetaFormer:
    mk = _recipe_kwargs([4, 4, 12, 4], [64, 128, 320, 512], IdentityMixing, **kwargs)
    return _create_metaformer("identityformer_s24", pretrained=pretrained, **mk)


@register_model
def identityformer_s36(pretrained: bool = False, **kwargs) -> MetaFormer:
    mk = _recipe_kwargs([6, 6, 18, 6], [64, 128, 320, 512], IdentityMixing, **kwargs)
    return _create_metaformer("identityformer_s36", pretrained=pretrained, **mk)


@register_model
def identityformer_m36(pretrained: bool = False, **kwargs) -> MetaFormer:
    mk = _recipe_kwargs([6, 6, 18, 6], [96, 192, 384, 768], IdentityMixing, **kwargs)
    return _create_metaformer("identityformer_m36", pretrained=pretrained, **mk)


@register_model
def identityformer_m48(pretrained: bool = False, **kwargs) -> MetaFormer:
    mk = _recipe_kwargs([8, 8, 24, 8], [96, 192, 384, 768], IdentityMixing, **kwargs)
    return _create_metaformer("identityformer_m48", pretrained=pretrained, **mk)


# -------------------------
# RandFormer (T = Id,Id,Rand,Rand)  -- fixed for 224 input
# -------------------------

@register_model
def randformer_s12(pretrained: bool = False, **kwargs) -> MetaFormer:
    token_mixers = [IdentityMixing, IdentityMixing, RandomMixing196, RandomMixing49]
    mk = _recipe_kwargs([2, 2, 6, 2], [64, 128, 320, 512], token_mixers, **kwargs)
    return _create_metaformer("randformer_s12", pretrained=pretrained, **mk)


@register_model
def randformer_s24(pretrained: bool = False, **kwargs) -> MetaFormer:
    token_mixers = [IdentityMixing, IdentityMixing, RandomMixing196, RandomMixing49]
    mk = _recipe_kwargs([4, 4, 12, 4], [64, 128, 320, 512], token_mixers, **kwargs)
    return _create_metaformer("randformer_s24", pretrained=pretrained, **mk)


@register_model
def randformer_s36(pretrained: bool = False, **kwargs) -> MetaFormer:
    token_mixers = [IdentityMixing, IdentityMixing, RandomMixing196, RandomMixing49]
    mk = _recipe_kwargs([6, 6, 18, 6], [64, 128, 320, 512], token_mixers, **kwargs)
    return _create_metaformer("randformer_s36", pretrained=pretrained, **mk)


@register_model
def randformer_m36(pretrained: bool = False, **kwargs) -> MetaFormer:
    token_mixers = [IdentityMixing, IdentityMixing, RandomMixing196, RandomMixing49]
    mk = _recipe_kwargs([6, 6, 18, 6], [96, 192, 384, 768], token_mixers, **kwargs)
    return _create_metaformer("randformer_m36", pretrained=pretrained, **mk)


@register_model
def randformer_m48(pretrained: bool = False, **kwargs) -> MetaFormer:
    token_mixers = [IdentityMixing, IdentityMixing, RandomMixing196, RandomMixing49]
    mk = _recipe_kwargs([8, 8, 24, 8], [96, 192, 384, 768], token_mixers, **kwargs)
    return _create_metaformer("randformer_m48", pretrained=pretrained, **mk)
