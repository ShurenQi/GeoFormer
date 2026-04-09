from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.layers import GroupNorm1
from timm.models._registry import register_model
from timm.models.metaformer import _create_metaformer


def _pct_kernel(k: int, n: int, m: int, trig: str = "cos") -> torch.Tensor:
    assert k % 2 == 1
    R = (k - 1) / 2.0
    coords = torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r = torch.sqrt(xx * xx + yy * yy)
    theta = torch.atan2(yy, xx)
    r_norm = r / R
    radial = torch.cos(math.pi * n * r_norm)
    mask = (r_norm <= (1.0 + 1e-6)).to(radial.dtype)
    radial = radial * mask
    if trig == "cos":
        angular = torch.cos(m * theta)
    else:
        angular = torch.sin(m * theta)
    W = radial * angular
    denom = torch.sum(torch.abs(W))
    W = torch.where(denom > 1e-6, W / denom, W)
    return W


class _FixedDepthwiseFilter(nn.Module):
    def __init__(self, channels: int, weight_2d: torch.Tensor, padding: str = "same", stride: int = 1):
        super().__init__()
        assert weight_2d.ndim == 2
        k_h, k_w = weight_2d.shape
        assert k_h == k_w and k_h % 2 == 1
        w = weight_2d.to(dtype=torch.float32).view(1, 1, k_h, k_w).repeat(int(channels), 1, 1, 1)
        self.register_buffer("weight", w, persistent=True)
        self.groups = int(channels)
        self.stride = int(stride)
        self.padding = (k_h // 2, k_w // 2) if padding == "same" else (0, 0)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)


class _GeometricBranchStride(nn.Module):
    def __init__(self, channels: int, k: int, n: int, m: int, stride: int = 1, eps: float = 1e-4):
        super().__init__()
        self.m = int(m)
        self.eps = float(eps)
        self.stride = int(stride)
        if self.m == 0:
            w = _pct_kernel(int(k), int(n), 0, trig="cos")
            self.op = _FixedDepthwiseFilter(int(channels), w, padding="same", stride=self.stride)
        else:
            w_c = _pct_kernel(int(k), int(n), int(m), trig="cos")
            w_s = _pct_kernel(int(k), int(n), int(m), trig="sin")
            self.op_c = _FixedDepthwiseFilter(int(channels), w_c, padding="same", stride=self.stride)
            self.op_s = _FixedDepthwiseFilter(int(channels), w_s, padding="same", stride=self.stride)

    def forward(self, x: Tensor) -> Tensor:
        if self.m == 0:
            return torch.abs(self.op(x))
        c = self.op_c(x)
        s = self.op_s(x)
        return torch.sqrt(c * c + s * s + self.eps)


def _orders_default_for_k(k: int) -> List[Tuple[int, int]]:
    k = int(k)
    if k == 3:
        return [(1, 0), (0, 1), (1, 1)]
    if k == 5:
        return [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    if k == 7:
        return [
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
        ]
    raise ValueError(f"Unsupported k={k}")


def _orders_rich_for_k(k: int) -> List[Tuple[int, int]]:
    k = int(k)
    if k == 3:
        return [(1, 0), (0, 1), (1, 1)]
    if k == 5:
        return [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    if k == 7:
        return [
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
            (1, 3),
            (2, 3),
            (3, 3),
        ]
    raise ValueError(f"Unsupported k={k}")


class GeometricTokenMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        kernel_size: int = 5,
        orders: Optional[List[Tuple[int, int]]] = None,
        min_ch_per_branch: int = 8,
        detail_alpha_init: float = 3e-3,
        use_bn: bool = False,
        eps: float = 1e-4,
        max_m_cap: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        assert self.kernel_size in (3, 5, 7) and (self.kernel_size % 2 == 1)
        if orders is None:
            orders = _orders_default_for_k(self.kernel_size)

        w_anchor = _pct_kernel(self.kernel_size, 0, 0, trig="cos")
        self.anchor = _FixedDepthwiseFilter(self.dim, w_anchor, padding="same")

        max_order = self.kernel_size // 2
        max_m = min(int(max_m_cap), max_order)
        configs: List[Tuple[int, int, int]] = []
        for (n, m) in orders:
            n = int(n)
            m = int(m)
            if n > max_order or m > max_m:
                continue
            if n == 0 and m == 0:
                continue
            configs.append((self.kernel_size, n, m))
        configs.sort(key=lambda t: (t[2], t[1], -t[0]))

        min_ch = max(1, int(min_ch_per_branch))
        max_branches = max(1, self.dim // min_ch)
        configs = configs[:max_branches]
        self.num_branches = max(1, len(configs))

        base = self.dim // self.num_branches
        rem = self.dim % self.num_branches
        self.split_sizes = [(base + (1 if i < rem else 0)) for i in range(self.num_branches)]
        assert sum(self.split_sizes) == self.dim

        self.detail_branches = nn.ModuleList()
        for i, (k, n, m) in enumerate(configs):
            ch_i = self.split_sizes[i]
            self.detail_branches.append(_GeometricBranchStride(ch_i, k=k, n=n, m=m, stride=1, eps=eps))

        self.detail_norm = nn.BatchNorm2d(self.dim) if use_bn else nn.Identity()
        self.alpha_detail = nn.Parameter(torch.ones(1, self.dim, 1, 1) * float(detail_alpha_init))
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop and proj_drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out_anchor = self.anchor(x) - x
        if len(self.detail_branches) == 0:
            return self.proj_drop(out_anchor)
        xs = torch.split(x, self.split_sizes, dim=1)
        detail_parts = [b(xi) for b, xi in zip(self.detail_branches, xs)]
        out_detail = torch.cat(detail_parts, dim=1)
        out_detail = self.detail_norm(out_detail)
        out = out_anchor + self.alpha_detail * out_detail
        return self.proj_drop(out)


class GeometricTokenMixerK5(GeometricTokenMixer):
    def __init__(self, dim: int, proj_drop: float = 0.0, **kwargs):
        super().__init__(
            dim=dim,
            proj_drop=proj_drop,
            kernel_size=5,
            use_bn=False,
            orders=[(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
            max_m_cap=2,
            **kwargs,
        )


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
def geometaformer_s12_k5(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([2, 2, 6, 2], [64, 128, 320, 512], GeometricTokenMixerK5, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s12_k5", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s24_k5(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([4, 4, 12, 4], [64, 128, 320, 512], GeometricTokenMixerK5, 1e-5, **kwargs)
    return _create_metaformer("geometaformer_s24_k5", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_s36_k5(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [64, 128, 320, 512], GeometricTokenMixerK5, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_s36_k5", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m36_k5(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([6, 6, 18, 6], [96, 192, 384, 768], GeometricTokenMixerK5, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m36_k5", pretrained=pretrained, **model_kwargs)


@register_model
def geometaformer_m48_k5(pretrained: bool = False, **kwargs):
    model_kwargs = _metaformer_kwargs([8, 8, 24, 8], [96, 192, 384, 768], GeometricTokenMixerK5, 1e-6, **kwargs)
    return _create_metaformer("geometaformer_m48_k5", pretrained=pretrained, **model_kwargs)
