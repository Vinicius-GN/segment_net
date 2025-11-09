# Source: Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (DeepLabv3+), ECCV 2018
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence

def _make_norm(num_channels: int, norm: str):
    if norm == "batch_norm":
        return nn.BatchNorm2d(num_channels, eps=1e-3, momentum=0.01)
    elif norm == "group_norm":
        num_groups = max(4, num_channels // 32)
        return nn.GroupNorm(num_groups, num_channels)
    else:
        return nn.Identity()

def _act(name: str):
    if name == "relu":
        return nn.ReLU(inplace=False)
    elif name == "silu":
        return nn.SiLU(inplace=False)
    elif name == "gelu":
        return nn.GELU()
    return nn.ReLU(inplace=False)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, norm: str = "batch_norm", act: str = "relu", groups: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=dilation, groups=groups, bias=False)
        self.bn = _make_norm(out_ch, norm)
        self.act = _act(act)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, d: int = 1, norm: str = "batch_norm", act: str = "relu"):
        super().__init__()
        p = d * (k // 2)
        self.dw = ConvBNAct(in_ch, in_ch, k=k, s=s, p=p, norm=norm, act=act, groups=in_ch, dilation=d)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, norm=norm, act=act)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return x

class ASPPBranch(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rate: int, norm: str = "batch_norm", act: str = "relu", separable: bool = True):
        super().__init__()
        if rate == 1:
            self.op = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, norm=norm, act=act)
        else:
            if separable:
                self.op = DepthwiseSeparableConv(in_ch, out_ch, k=3, s=1, d=rate, norm=norm, act=act)
            else:
                p = rate
                self.op = ConvBNAct(in_ch, out_ch, k=3, s=1, p=p, norm=norm, act=act, dilation=rate)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch_norm", act: str = "relu"):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, norm=norm, act=act)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y = self.pool(x)
        y = self.proj(y)
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return y

class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: Sequence[int], norm: str = "batch_norm", act: str = "relu", separable: bool = True, dropout: float = 0.0):
        super().__init__()
        branch_ch = out_ch
        self.branches = nn.ModuleList()
        self.branches.append(ASPPBranch(in_ch, branch_ch, rate=1, norm=norm, act=act, separable=False))
        for r in rates:
            self.branches.append(ASPPBranch(in_ch, branch_ch, rate=int(r), norm=norm, act=act, separable=separable))
        self.image_pool = ASPPPooling(in_ch, branch_ch, norm=norm, act=act)
        concat_ch = branch_ch * (len(rates) + 2)
        self.project = ConvBNAct(concat_ch, out_ch, k=1, s=1, p=0, norm=norm, act=act)
        self.drop = nn.Dropout2d(p=dropout, inplace=False) if dropout and dropout > 0 else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [b(x) for b in self.branches]
        ys.append(self.image_pool(x))
        y = torch.cat(ys, dim=1)
        y = self.project(y)
        y = self.drop(y)
        return y

class ASPPFusion(nn.Module):
    def __init__(self, channels: int, out_channels: int, num_levels: int, rates: Sequence[int], norm: str = "batch_norm", act: str = "relu", separable: bool = True, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([ASPP(channels, out_channels, rates=rates, norm=norm, act=act, separable=separable, dropout=dropout) for _ in range(num_levels)])
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for i, x in enumerate(inputs):
            outs.append(self.blocks[i](x))
        return outs
