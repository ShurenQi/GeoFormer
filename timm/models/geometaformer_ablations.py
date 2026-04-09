from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from timm.layers import GroupNorm1
from timm.models._registry import register_model
from timm.models.metaformer import _create_metaformer

from .geometaformer import GeometricTokenMixer, _orders_default_for_k, _orders_rich_for_k


def _norm2d(kind: str, channels: int) -> nn.Module:
    kind = str(kind).lower()
    if kind in ("bn", "batchnorm", "batchnorm2d"):
        return nn.BatchNorm2d(channels)
    if kind in ("gn", "groupnorm"):
        return nn.GroupNorm(1, channels)
    if kind in ("id", "identity", "none", ""):
        return nn.Identity()
    raise ValueError(f"Unsupported norm kind: {kind}")


class _ConvNormAct2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn", act: str = "gelu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=(norm in ("id", "identity", "none", "")))
        self.norm = _norm2d(norm, out_ch)
        if act.lower() == "gelu":
            self.act = nn.GELU()
        elif act.lower() in ("relu",):
            self.act = nn.ReLU(inplace=True)
        elif act.lower() in ("silu", "swish"):
            self.act = nn.SiLU(inplace=True)
        elif act.lower() in ("id", "identity", "none", ""):
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported act: {act}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class GeometricTokenMixerK3Abl(GeometricTokenMixer):
    def __init__(self, dim: int, proj_drop: float = 0.0, **kwargs):
        kwargs.pop("use_bn", None)
        super().__init__(
            dim=dim,
            proj_drop=proj_drop,
            kernel_size=3,
            use_bn=False,
            orders=_orders_default_for_k(3),
            max_m_cap=1,
            **kwargs,
        )


class GeometricTokenMixerK5Abl(GeometricTokenMixer):
    def __init__(self, dim: int, proj_drop: float = 0.0, **kwargs):
        kwargs.pop("use_bn", None)
        super().__init__(
            dim=dim,
            proj_drop=proj_drop,
            kernel_size=5,
            use_bn=False,
            orders=_orders_default_for_k(5),
            max_m_cap=2,
            **kwargs,
        )


class GeometricTokenMixerK7Abl(GeometricTokenMixer):
    def __init__(self, dim: int, proj_drop: float = 0.0, **kwargs):
        kwargs.pop("use_bn", None)
        super().__init__(
            dim=dim,
            proj_drop=proj_drop,
            kernel_size=7,
            use_bn=False,
            orders=_orders_default_for_k(7),
            max_m_cap=2,
            **kwargs,
        )


class GeometricTokenMixerK7HOAbl(GeometricTokenMixer):
    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        fuse_norm: str = "bn",
        **kwargs,
    ):
        kwargs.pop("use_bn", None)
        super().__init__(
            dim=dim,
            proj_drop=proj_drop,
            kernel_size=7,
            use_bn=False,
            orders=_orders_rich_for_k(7),
            max_m_cap=3,
            **kwargs,
        )
        self.detail_fuse = _ConvNormAct2d(self.dim, self.dim, norm=fuse_norm, act="gelu")

    def forward(self, x: Tensor) -> Tensor:
        out_anchor = self.anchor(x) - x
        if len(self.detail_branches) == 0:
            return self.proj_drop(out_anchor)

        xs = torch.split(x, self.split_sizes, dim=1)
        detail_parts = [b(xi) for b, xi in zip(self.detail_branches, xs)]
        out_detail = torch.cat(detail_parts, dim=1)
        out_detail = self.detail_norm(out_detail)
        out_detail = self.detail_fuse(out_detail)

        out = out_anchor + self.alpha_detail * out_detail
        return self.proj_drop(out)


class MultiScaleGeometricTokenMixerA1Abl(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        kernel_sizes: Tuple[int, ...] = (5, 7),
        orders_list: Optional[List[Optional[List[Tuple[int, int]]]]] = None,
        min_ch_per_branch: int = 8,
        detail_alpha_init: float = 3e-3,
        use_bn: bool = False,
        eps: float = 1e-4,
        max_m_cap: int = 2,
        fuse_norm: str = "bn",
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("use_bn", None)
        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        assert len(kernel_sizes) >= 2
        assert all(k in (3, 5, 7) and k % 2 == 1 for k in kernel_sizes)

        if orders_list is None:
            orders_list = [None] * len(kernel_sizes)
        assert len(orders_list) == len(kernel_sizes)

        self.num_scales = len(kernel_sizes)
        self.mixers = nn.ModuleList(
            [
                GeometricTokenMixer(
                    dim=dim,
                    proj_drop=0.0,
                    kernel_size=k,
                    orders=o,
                    min_ch_per_branch=min_ch_per_branch,
                    detail_alpha_init=detail_alpha_init,
                    use_bn=use_bn,
                    eps=eps,
                    max_m_cap=max_m_cap,
                    **kwargs,
                )
                for k, o in zip(kernel_sizes, orders_list)
            ]
        )
        self.fuse = _ConvNormAct2d(dim * self.num_scales, dim, norm=fuse_norm, act="gelu")
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop and proj_drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        ys = [m(x) for m in self.mixers]
        y = torch.cat(ys, dim=1)
        y = self.fuse(y)
        return self.proj_drop(y)


class MultiScaleGeometricTokenMixerK57A1Abl(MultiScaleGeometricTokenMixerA1Abl):
    def __init__(self, dim: int, proj_drop: float = 0.0, **kwargs):
        super().__init__(
            dim=dim,
            proj_drop=proj_drop,
            kernel_sizes=(5, 7),
            orders_list=[_orders_default_for_k(5), _orders_default_for_k(7)],
            max_m_cap=2,
            **kwargs,
        )


class MultiScaleGeometricTokenMixerK357A1Abl(MultiScaleGeometricTokenMixerA1Abl):
    def __init__(self, dim: int, proj_drop: float = 0.0, **kwargs):
        super().__init__(
            dim=dim,
            proj_drop=proj_drop,
            kernel_sizes=(3, 5, 7),
            orders_list=[_orders_default_for_k(3), _orders_default_for_k(5), _orders_default_for_k(7)],
            max_m_cap=2,
            **kwargs,
        )


class MultiScaleGeometricTokenMixerK57HOA1Abl(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        min_ch_per_branch: int = 8,
        detail_alpha_init: float = 3e-3,
        use_bn: bool = False,
        eps: float = 1e-4,
        fuse_norm: str = "bn",
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("use_bn", None)
        self.m5 = GeometricTokenMixer(
            dim=dim,
            proj_drop=0.0,
            kernel_size=5,
            orders=_orders_default_for_k(5),
            min_ch_per_branch=min_ch_per_branch,
            detail_alpha_init=detail_alpha_init,
            use_bn=use_bn,
            eps=eps,
            max_m_cap=2,
            **kwargs,
        )
        self.m7ho = GeometricTokenMixerK7HOAbl(
            dim=dim,
            proj_drop=0.0,
            fuse_norm=fuse_norm,
            min_ch_per_branch=min_ch_per_branch,
            detail_alpha_init=detail_alpha_init,
            use_bn=use_bn,
            eps=eps,
            **kwargs,
        )
        self.fuse = _ConvNormAct2d(dim * 2, dim, norm=fuse_norm, act="gelu")
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop and proj_drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y5 = self.m5(x)
        y7 = self.m7ho(x)
        y = torch.cat([y5, y7], dim=1)
        y = self.fuse(y)
        return self.proj_drop(y)


def _metaformer_kwargs(depths, dims, token_mixers, layer_scale_init_values: float, **kwargs):
    return dict(
        depths=depths,
        dims=dims,
        mlp_act=nn.GELU,
        mlp_bias=True,
        norm_layers=GroupNorm1,
        layer_scale_init_values=layer_scale_init_values,
        res_scale_init_values=None,
        use_mlp_head=False,
        token_mixers=token_mixers,
        **kwargs,
    )


@register_model
def geometaformer_s12_k3_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([2, 2, 6, 2], [64, 128, 320, 512], GeometricTokenMixerK3Abl, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s12_k3_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s24_k3_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([4, 4, 12, 4], [64, 128, 320, 512], GeometricTokenMixerK3Abl, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s24_k3_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s36_k3_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [64, 128, 320, 512], GeometricTokenMixerK3Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_s36_k3_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m36_k3_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [96, 192, 384, 768], GeometricTokenMixerK3Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m36_k3_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m48_k3_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([8, 8, 24, 8], [96, 192, 384, 768], GeometricTokenMixerK3Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m48_k3_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s12_k5_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([2, 2, 6, 2], [64, 128, 320, 512], GeometricTokenMixerK5Abl, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s12_k5_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s24_k5_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([4, 4, 12, 4], [64, 128, 320, 512], GeometricTokenMixerK5Abl, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s24_k5_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s36_k5_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [64, 128, 320, 512], GeometricTokenMixerK5Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_s36_k5_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m36_k5_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [96, 192, 384, 768], GeometricTokenMixerK5Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m36_k5_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m48_k5_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([8, 8, 24, 8], [96, 192, 384, 768], GeometricTokenMixerK5Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m48_k5_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s12_k7_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([2, 2, 6, 2], [64, 128, 320, 512], GeometricTokenMixerK7Abl, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s12_k7_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s24_k7_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([4, 4, 12, 4], [64, 128, 320, 512], GeometricTokenMixerK7Abl, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s24_k7_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s36_k7_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [64, 128, 320, 512], GeometricTokenMixerK7Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_s36_k7_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m36_k7_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [96, 192, 384, 768], GeometricTokenMixerK7Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m36_k7_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m48_k7_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([8, 8, 24, 8], [96, 192, 384, 768], GeometricTokenMixerK7Abl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m48_k7_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s12_k57_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [2, 2, 6, 2], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK57A1Abl, 1e-5, **kwargs
    )
    return _create_metaformer("geometaformer_s12_k57_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s24_k57_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [4, 4, 12, 4], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK57A1Abl, 1e-5, **kwargs
    )
    return _create_metaformer("geometaformer_s24_k57_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s36_k57_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [6, 6, 18, 6], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK57A1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_s36_k57_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m36_k57_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [6, 6, 18, 6], [96, 192, 384, 768], MultiScaleGeometricTokenMixerK57A1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_m36_k57_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m48_k57_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [8, 8, 24, 8], [96, 192, 384, 768], MultiScaleGeometricTokenMixerK57A1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_m48_k57_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s12_k357_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [2, 2, 6, 2], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK357A1Abl, 1e-5, **kwargs
    )
    return _create_metaformer("geometaformer_s12_k357_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s24_k357_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [4, 4, 12, 4], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK357A1Abl, 1e-5, **kwargs
    )
    return _create_metaformer("geometaformer_s24_k357_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s36_k357_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [6, 6, 18, 6], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK357A1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_s36_k357_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m36_k357_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [6, 6, 18, 6], [96, 192, 384, 768], MultiScaleGeometricTokenMixerK357A1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_m36_k357_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m48_k357_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [8, 8, 24, 8], [96, 192, 384, 768], MultiScaleGeometricTokenMixerK357A1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_m48_k357_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s12_k7_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([2, 2, 6, 2], [64, 128, 320, 512], GeometricTokenMixerK7HOAbl, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s12_k7_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s24_k7_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([4, 4, 12, 4], [64, 128, 320, 512], GeometricTokenMixerK7HOAbl, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s24_k7_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s36_k7_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [64, 128, 320, 512], GeometricTokenMixerK7HOAbl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_s36_k7_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m36_k7_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [96, 192, 384, 768], GeometricTokenMixerK7HOAbl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m36_k7_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m48_k7_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([8, 8, 24, 8], [96, 192, 384, 768], GeometricTokenMixerK7HOAbl, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m48_k7_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s12_k57_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [2, 2, 6, 2], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK57HOA1Abl, 1e-5, **kwargs
    )
    return _create_metaformer("geometaformer_s12_k57_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s24_k57_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [4, 4, 12, 4], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK57HOA1Abl, 1e-5, **kwargs
    )
    return _create_metaformer("geometaformer_s24_k57_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s36_k57_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [6, 6, 18, 6], [64, 128, 320, 512], MultiScaleGeometricTokenMixerK57HOA1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_s36_k57_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m36_k57_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [6, 6, 18, 6], [96, 192, 384, 768], MultiScaleGeometricTokenMixerK57HOA1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_m36_k57_ho_abl", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m48_k57_ho_abl(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs(
        [8, 8, 24, 8], [96, 192, 384, 768], MultiScaleGeometricTokenMixerK57HOA1Abl, 1e-6, **kwargs
    )
    return _create_metaformer("geometaformer_m48_k57_ho_abl", pretrained=pretrained, **model_kwargs)
