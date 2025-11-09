# Source: Liu et al., "Path Aggregation Network for Instance Segmentation", CVPR 2018
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def _make_norm(num_channels: int, norm: str):
    if norm == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    elif norm == "group_norm":
        num_groups = max(4, num_channels // 32)
        return nn.GroupNorm(num_groups, num_channels)
    else:
        return nn.Identity()

def _make_act():
    return nn.ReLU(inplace=True)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, norm: str):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = _make_norm(out_ch, norm)
        self.act = _make_act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class PANet(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int, norm: str = "batch_norm"):
        super().__init__()
        self.L = len(in_channels_list)
        self.proj = nn.ModuleList([
            ConvBNAct(c, out_channels, k=1, s=1, p=0, norm=norm) if c != out_channels else nn.Sequential()
            for c in in_channels_list
        ])
        self.down_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, k=3, s=2, p=1, norm=norm) for _ in range(self.L - 1)
        ])
        self.smooth_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, k=3, s=1, p=1, norm=norm) for _ in range(self.L - 1)
        ])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        feats = []
        for i, f in enumerate(inputs):
            x = f if isinstance(self.proj[i], nn.Sequential) else self.proj[i](f)
            feats.append(x)
        outs = [None] * self.L
        outs[0] = feats[0]
        for i in range(self.L - 1):
            down = self.down_convs[i](outs[i])
            fused = down + feats[i + 1]
            outs[i + 1] = self.smooth_convs[i](fused)
        return outs
