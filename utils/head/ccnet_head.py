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


class CrissCrossAttention(nn.Module):
    """Pure-PyTorch CCA, rows+cols attention; shared across RCCA steps."""
    def __init__(self, in_channels: int, inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(8, in_channels // 8)
        self.query = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.key   = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels,   kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _attend_rows(q, k, v):
        B, Cq, H, W = q.shape
        Cv = v.shape[1]
        qr = q.permute(0, 2, 3, 1).reshape(B * H, W, Cq)
        kr = k.permute(0, 2, 3, 1).reshape(B * H, W, Cq)
        vr = v.permute(0, 2, 3, 1).reshape(B * H, W, Cv)
        energy = torch.bmm(qr, kr.transpose(1, 2))  # (B*H, W, W)
        eye = torch.eye(W, device=energy.device, dtype=energy.dtype).unsqueeze(0).repeat(B * H, 1, 1)
        energy = energy.masked_fill_(eye.bool(), float("-inf"))
        attn = F.softmax(energy, dim=-1)
        out = torch.bmm(attn, vr).reshape(B, H, W, Cv).permute(0, 3, 1, 2)
        return out

    @staticmethod
    def _attend_cols(q, k, v):
        B, Cq, H, W = q.shape
        Cv = v.shape[1]
        qc = q.permute(0, 3, 2, 1).reshape(B * W, H, Cq)
        kc = k.permute(0, 3, 2, 1).reshape(B * W, H, Cq)
        vc = v.permute(0, 3, 2, 1).reshape(B * W, H, Cv)
        energy = torch.bmm(qc, kc.transpose(1, 2))  # (B*W, H, H)
        eye = torch.eye(H, device=energy.device, dtype=energy.dtype).unsqueeze(0).repeat(B * W, 1, 1)
        energy = energy.masked_fill_(eye.bool(), float("-inf"))
        attn = F.softmax(energy, dim=-1)
        out = torch.bmm(attn, vc).reshape(B, W, H, Cv).permute(0, 3, 2, 1).contiguous()
        return out

    def forward(self, x: torch.Tensor):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        out_r = self._attend_rows(q, k, v)
        out_c = self._attend_cols(q, k, v)
        out = out_r + out_c
        return self.gamma * out + x


class RCCABlock(nn.Module):
    def __init__(self, channels: int, inter_channels: int = None, recurrence: int = 2):
        super().__init__()
        self.recurrence = max(1, int(recurrence))
        self.cca = CrissCrossAttention(channels, inter_channels)

    def forward(self, x: torch.Tensor):
        y = x
        for _ in range(self.recurrence):
            y = self.cca(y)
        return y


class CCNetHead(nn.Module):
    """
    CCNet-style head with a 3x3 fusion, 1-2 RCCA steps, and a classifier.
    Paper: 'CCNet: Criss-Cross Attention for Semantic Segmentation' (ICCV'19).
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

        self.recurrence = int(self.config.get("head").get("rcca_recurrence", 2))
        self.reduction  = int(self.config.get("head").get("rcca_reduction", 8))
        self.mid_kernel = int(self.config.get("head").get("fusion_kernel", 3))

        self._built = False
        self._in_ch = None

        self.fuse = None
        self.fuse_norm = None
        self.act = nn.ReLU(inplace=True)
        self.rcca = None
        self.dropout = nn.Dropout(self.dropout_prob, inplace=False)
        self.classifier = None

    def _maybe_build(self, in_ch: int, device, dtype):
        if self._built and self._in_ch == in_ch:
            return
        pad = (self.mid_kernel - 1) // 2
        self.fuse = nn.Conv2d(in_ch, in_ch, kernel_size=self.mid_kernel, padding=pad, bias=False).to(device=device, dtype=dtype)
        self.fuse_norm = _make_norm(self.norm_fn, in_ch).to(device=device, dtype=dtype)
        inter = max(8, in_ch // max(1, self.reduction))
        self.rcca = RCCABlock(in_ch, inter_channels=inter, recurrence=self.recurrence).to(device=device, dtype=dtype)
        self.classifier = nn.Conv2d(in_ch, self.num_classes, kernel_size=1).to(device=device, dtype=dtype)
        self._in_ch = in_ch
        self._built = True

    @staticmethod
    def _ensure_list(x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        return _ensure_list_features(x)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        feats = self._ensure_list(x)
        x_in = _pick_feature(feats, self.aggregate)
        self._maybe_build(int(x_in.shape[1]), x_in.device, x_in.dtype)

        z = self.act(self.fuse_norm(self.fuse(x_in)))
        z = self.rcca(z)
        z = self.dropout(z)

        if self.opt_latency:
            z = self.classifier(z)
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
        else:
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
            z = self.classifier(z)
        return z
