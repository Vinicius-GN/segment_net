import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def make_norm(norm_fn: str, num_channels: int) -> nn.Module:
    if norm_fn == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    if norm_fn == "group_norm":
        groups = max(2, num_channels // 16)
        return nn.GroupNorm(groups, num_channels)
    return nn.Identity()

class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: List[int], norm_fn: str, dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList()

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            make_norm(norm_fn, out_ch),
            nn.ReLU(inplace=True)
        ))

        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                make_norm(norm_fn, out_ch),
                nn.ReLU(inplace=True)
            ))

        self.pool_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pool_norm = make_norm(norm_fn, out_ch)
        self.pool_act  = nn.ReLU(inplace=True)

        proj_in = out_ch * (2 + len(rates))  # 1x1 + len(rates) + pooling
        self.project = nn.Sequential(
            nn.Conv2d(proj_in, out_ch, kernel_size=1, bias=False),
            make_norm(norm_fn, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        feats = [b(x) for b in self.branches]

        pooled = F.adaptive_avg_pool2d(x, 1)
        pooled = self.pool_conv(pooled)
        pooled = self.pool_norm(pooled)
        pooled = self.pool_act(pooled)
        pooled = F.interpolate(pooled, size=(H, W), mode="bilinear", align_corners=False)
        feats.append(pooled)

        y = torch.cat(feats, dim=1)
        return self.project(y)


class DeepLabV3PlusCore(nn.Module):

    def __init__(
        self,
        in_ch_high: int,             
        in_ch_low:  int,                
        aspp_channels: int = 256,
        decoder_channels: int = 256,
        low_channels: int = 48,
        aspp_rates: List[int] = (6, 12, 18),
        norm_fn: str = "batch_norm",
        dropout: float = 0.1,
    ):
        super().__init__()

        self._cfg = {
            "aspp_channels": aspp_channels,
            "aspp_rates": list(aspp_rates),
            "norm_fn": norm_fn,
            "dropout": dropout,
            "low_channels": low_channels,
        }

        self.aspp = None
        self.low_proj = None
        self._aspp_in_ch = None
        self._low_in_ch  = None

        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels + low_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            make_norm(norm_fn, decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            make_norm(norm_fn, decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self._build_aspp(in_ch_high)
        self._build_low_proj(in_ch_low)

    def _build_aspp(self, in_ch: int, device=None, dtype=None):
        self.aspp = ASPP(
            in_ch=in_ch,
            out_ch=self._cfg["aspp_channels"],
            rates=self._cfg["aspp_rates"],
            norm_fn=self._cfg["norm_fn"],
            dropout=self._cfg["dropout"]
        )
        if device is not None:
            self.aspp.to(device=device, dtype=dtype)
        self._aspp_in_ch = in_ch

    def _build_low_proj(self, in_ch: int, device=None, dtype=None):
        self.low_proj = nn.Sequential(
            nn.Conv2d(in_ch, self._cfg["low_channels"], kernel_size=1, bias=False),
            make_norm(self._cfg["norm_fn"], self._cfg["low_channels"]),
            nn.ReLU(inplace=True)
        )
        if device is not None:
            self.low_proj.to(device=device, dtype=dtype)
        self._low_in_ch = in_ch

    def _maybe_rebuild_aspp(self, x: torch.Tensor):
        in_ch = x.shape[1]
        if (self.aspp is None) or (in_ch != self._aspp_in_ch):
            self._build_aspp(in_ch, device=x.device, dtype=x.dtype)

    def _maybe_rebuild_low_proj(self, low: torch.Tensor):
        in_ch = low.shape[1]
        if (self.low_proj is None) or (in_ch != self._low_in_ch):
            self._build_low_proj(in_ch, device=low.device, dtype=low.dtype)

    @staticmethod
    def pick_high_low(feats: List[torch.Tensor]):
        sizes = [f.shape[-2:] for f in feats]
        hi = min(range(len(feats)), key=lambda i: sizes[i][0] * sizes[i][1])
        lo = max(range(len(feats)), key=lambda i: sizes[i][0] * sizes[i][1])
        return feats[hi], feats[lo]

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_rebuild_aspp(x)
        return self.aspp(x)

    def forward_pair(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        self._maybe_rebuild_aspp(high)
        self._maybe_rebuild_low_proj(low)
        y_aspp = self.aspp(high)
        y_aspp = F.interpolate(y_aspp, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y_low  = self.low_proj(low)
        y = torch.cat([y_aspp, y_low], dim=1)
        return self.decoder(y)
