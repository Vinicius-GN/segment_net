import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple


class _IdentityNorm(nn.Module):
    def forward(self, x):
        return x


def _make_norm(kind: str, num_channels: int) -> nn.Module:
    if kind == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    if kind == "group_norm":
        groups = max(2, num_channels // 16)
        return nn.GroupNorm(groups, num_channels)
    return _IdentityNorm()


def _ensure_list_features(x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _pick_feature(feats: List[torch.Tensor], aggregate: str) -> torch.Tensor:
    if aggregate == "concat":
        return torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
    if len(feats) == 1:
        return feats[0]
    return min(feats, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))  # lowest-res map


class _DenseASPPLayer(nn.Module):
    """1x1 reduce -> 3x3 atrous -> growth; concat happens in the caller."""
    def __init__(self, in_channels: int, inter_channels: int, growth_rate: int, dilation: int, norm_fn: str):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.reduce_norm = _make_norm(norm_fn, inter_channels)
        self.conv = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.conv_norm = _make_norm(norm_fn, growth_rate)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.reduce_norm(self.reduce(x)))
        y = self.act(self.conv_norm(self.conv(y)))
        return y  # growth feature


class DenseASPPHead(nn.Module):
    """
    DenseASPP decode head with dense atrous blocks and final 1x1 + classifier.
    Paper: 'DenseASPP for Semantic Segmentation in Street Scenes' (CVPR'18).
    """

    def __init__(self, config, num_classes: int, num_feature_layers: int):
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        self.num_feature_layers = int(num_feature_layers)

        self.image_size = list(reversed(self.config.get("image").get("image_size")))
        self.opt_latency = bool(self.config.get("head").get("opt_latency"))
        self.dropout_prob = float(self.config.get("head").get("dropout", 0.0))
        self.norm_fn = (self.config.get("model").get("norm_fn") or "batch_norm")
        self.aggregate = (self.config.get("backbone").get("aggregate") or "").lower()

        self.rates = tuple(self.config.get("head").get("denseaspp_rates", (3, 6, 12, 18, 24)))
        self.inter_channels_hint = int(self.config.get("head").get("denseaspp_inter_channels", 256))
        self.growth_rate_hint = int(self.config.get("head").get("denseaspp_growth_rate", 0))
        self.out_channels_hint = int(self.config.get("head").get("denseaspp_out_channels", 0))

        self.dropout = nn.Dropout(self.dropout_prob, inplace=False)

        self._built = False
        self._in_ch = None
        self.blocks = None
        self.fuse = None
        self.fuse_norm = None
        self.classifier = None
        self.act = nn.ReLU(inplace=True)

    def _maybe_build(self, in_ch: int, device, dtype):
        if self._built and self._in_ch == in_ch:
            return

        growth_rate = self.growth_rate_hint if self.growth_rate_hint > 0 else max(8, in_ch // 8)
        inter_channels = self.inter_channels_hint
        cur_in = in_ch
        layers = []
        for d in self.rates:
            layers.append(_DenseASPPLayer(cur_in, inter_channels, growth_rate, d, self.norm_fn))
            cur_in = cur_in + growth_rate  # dense concat
        self.blocks = nn.ModuleList(layers).to(device=device, dtype=dtype)

        out_ch = self.out_channels_hint if self.out_channels_hint > 0 else in_ch
        total_ch = in_ch + len(self.rates) * growth_rate
        self.fuse = nn.Conv2d(total_ch, out_ch, kernel_size=1, bias=False).to(device=device, dtype=dtype)
        self.fuse_norm = _make_norm(self.norm_fn, out_ch).to(device=device, dtype=dtype)
        self.classifier = nn.Conv2d(out_ch, self.num_classes, kernel_size=1).to(device=device, dtype=dtype)

        self._in_ch = in_ch
        self._built = True

    @staticmethod
    def _ensure_list(x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        return _ensure_list_features(x)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        feats = self._ensure_list(x)
        x_in = _pick_feature(feats, self.aggregate)
        self._maybe_build(int(x_in.shape[1]), x_in.device, x_in.dtype)

        feats_cat = [x_in]
        cur = x_in
        for blk in self.blocks:
            y = blk(torch.cat(feats_cat, dim=1))
            feats_cat.append(y)
        z = torch.cat(feats_cat, dim=1)
        z = self.act(self.fuse_norm(self.fuse(z)))
        z = self.dropout(z)

        if self.opt_latency:
            z = self.classifier(z)
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
        else:
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
            z = self.classifier(z)
        return z
