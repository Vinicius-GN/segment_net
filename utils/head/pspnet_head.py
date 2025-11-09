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
    return min(feats, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))


class PyramidPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool_scales=(1, 2, 3, 6), norm_fn="batch_norm"):
        super().__init__()
        self.branches = nn.ModuleList()
        for s in pool_scales:
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=s),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    _make_norm(norm_fn, out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.pool_scales = tuple(pool_scales)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h, w = x.shape[-2:]
        outs = [x]
        for m in self.branches:
            y = m(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(y)
        return outs  # [x, p1, p2, p3, p6]


class PSPNetHead(nn.Module):
    """
    PSPNet-style head with Pyramid Pooling Module.

    Config keys used:
      - image.image_size
      - image.num_classes (passed as ctor arg)
      - head.opt_latency (bool)
      - head.dropout (float)
      - head.ppm_pool_scales (optional, default (1,2,3,6))
      - model.norm_fn in {"batch_norm","group_norm"} else Identity
      - backbone.fpn_out_channels (hint)
      - backbone.aggregate in {"", "none", "concat"}
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
        self.pool_scales = tuple(self.config.get("head").get("ppm_pool_scales", (1, 2, 3, 6)))

        self._built = False
        self._in_ch = None

        self.ppm = None
        self.bottleneck = None
        self.bottleneck_norm = None
        self.dropout = nn.Dropout(self.dropout_prob, inplace=False)
        self.classifier = None

    def _maybe_build(self, in_ch: int, device, dtype):
        if self._built and self._in_ch == in_ch:
            return

        out_each = max(16, in_ch // max(1, len(self.pool_scales)))
        self.ppm = PyramidPooling(in_ch, out_each, pool_scales=self.pool_scales, norm_fn=self.norm_fn)
        self.ppm.to(device=device, dtype=dtype)

        concat_ch = in_ch + len(self.pool_scales) * out_each
        bottleneck_ch = max(32, in_ch // 2)
        self.bottleneck = nn.Conv2d(concat_ch, bottleneck_ch, kernel_size=3, padding=1, bias=False).to(device=device, dtype=dtype)
        self.bottleneck_norm = _make_norm(self.norm_fn, bottleneck_ch).to(device=device, dtype=dtype)

        self.classifier = nn.Conv2d(bottleneck_ch, self.num_classes, kernel_size=1).to(device=device, dtype=dtype)

        self._in_ch = in_ch
        self._built = True

    @staticmethod
    def _ensure_list(x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        return _ensure_list_features(x)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        feats = self._ensure_list(x)
        x_in = _pick_feature(feats, self.aggregate) if self.aggregate != "concat" else torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]

        self._maybe_build(int(x_in.shape[1]), x_in.device, x_in.dtype)

        outs = self.ppm(x_in)
        z = torch.cat(outs, dim=1)
        z = self.bottleneck_norm(self.bottleneck(z))
        z = F.relu(z, inplace=True)
        z = self.dropout(z)

        if self.opt_latency:
            z = self.classifier(z)
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
        else:
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
            z = self.classifier(z)
        return z
