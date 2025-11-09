# Source: Wang et al., "Deep High-Resolution Representation Learning for Visual Recognition" (HRNet, TPAMI 2020)
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

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, norm: str, act: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = _make_norm(out_ch, norm)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Downsample2x(nn.Module):
    def __init__(self, channels: int, norm: str):
        super().__init__()
        self.op = ConvBNAct(channels, channels, k=3, s=2, p=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class Upsample1x1(nn.Module):
    def __init__(self, channels: int, norm: str, act: str = "relu"):
        super().__init__()
        self.proj = ConvBNAct(channels, channels, k=1, s=1, p=0, norm=norm, act=act)

    def forward(self, x: torch.Tensor, size_hw) -> torch.Tensor:
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        x = self.proj(x)
        return x

class ExchangeUnit(nn.Module):
    def __init__(self, num_branches: int, channels: int, norm: str = "batch_norm"):
        super().__init__()
        self.B = num_branches
        self.channels = channels
        self.norm = norm
        self.down_ops = nn.ModuleDict()
        self.up_ops = nn.ModuleDict()
        for x in range(self.B):
            for r in range(self.B):
                key = f"{x}->{r}"
                if x == r:
                    self.down_ops[key] = nn.Identity()
                    self.up_ops[key] = nn.Identity()
                elif x < r:
                    seq = []
                    for _ in range(r - x):
                        seq.append(Downsample2x(channels, norm))
                    self.down_ops[key] = nn.Sequential(*seq)
                    self.up_ops[key] = nn.Identity()
                else:
                    self.down_ops[key] = nn.Identity()
                    self.up_ops[key] = Upsample1x1(channels, norm)
        self.refine = nn.ModuleList([ConvBNAct(channels, channels, k=3, s=1, p=1, norm=norm) for _ in range(self.B)])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(inputs) == self.B
        sizes = [t.shape[-2:] for t in inputs]
        outs = []
        for r in range(self.B):
            acc = 0
            for x in range(self.B):
                t = inputs[x]
                if x < r:
                    t = self.down_ops[f"{x}->{r}"](t)
                elif x > r:
                    t = self.up_ops[f"{x}->{r}"](t, sizes[r])
                acc = acc + t
            outs.append(self.refine[r](acc))
        return outs

class HRNetFusion(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int, num_modules: int = 2, norm: str = "batch_norm", act: str = "relu"):
        super().__init__()
        self.B = len(in_channels_list)
        self.proj = nn.ModuleList([
            ConvBNAct(c, out_channels, k=1, s=1, p=0, norm=norm, act=act) if c != out_channels else nn.Identity()
            for c in in_channels_list
        ])
        self.modules_list = nn.ModuleList([ExchangeUnit(self.B, out_channels, norm=norm) for _ in range(num_modules)])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(inputs) == self.B
        x = []
        for i in range(self.B):
            xi = inputs[i] if isinstance(self.proj[i], nn.Identity) else self.proj[i](inputs[i])
            x.append(xi)
        for exch in self.modules_list:
            x = exch(x)
        return x
