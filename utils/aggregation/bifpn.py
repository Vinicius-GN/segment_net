# Source: Tan, Pang, Le. "EfficientDet: Scalable and Efficient Object Detection" (BiFPN)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Optional

class SeparableConvBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "bn", act: str = "silu"):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, stride=1, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        if norm == "bn":
            self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        elif norm == "gn":
            ng = max(4, channels // 32)
            self.bn = nn.GroupNorm(ng, channels)
        else:
            self.bn = nn.Identity()
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Conv1x1BnAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn", act: str = "silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        if norm == "bn":
            self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        elif norm == "gn":
            ng = max(4, out_ch // 32)
            self.bn = nn.GroupNorm(ng, out_ch)
        else:
            self.bn = nn.Identity()
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

def _resize_to(x: torch.Tensor, size_hw: Sequence[int]) -> torch.Tensor:
    if list(x.shape[-2:]) == list(size_hw):
        return x
    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

class FastWeightedSum(nn.Module):
    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        w = torch.relu(self.w)
        weight = w / (w.sum() + self.eps)
        x = 0.0
        for wi, ti in zip(weight, inputs):
            x = x + wi * ti
        return x

class BiFPNLayer(nn.Module):
    def __init__(self, channels: int, n_levels: int, norm: str = "bn", act: str = "silu"):
        super().__init__()
        self.n_levels = n_levels
        self.epsilon = 1e-4
        self.top_down_convs = nn.ModuleList([SeparableConvBlock(channels, norm, act) for _ in range(n_levels)])
        self.bottom_up_convs = nn.ModuleList([SeparableConvBlock(channels, norm, act) for _ in range(n_levels)])
        self.td_weights = nn.ModuleList([FastWeightedSum(2) for _ in range(n_levels-1)])
        self.out_weights_first = FastWeightedSum(2)
        self.out_weights = nn.ModuleList([FastWeightedSum(3) for _ in range(n_levels-1)])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        n = self.n_levels
        td = [None] * n
        td[-1] = self.top_down_convs[-1](feats[-1])
        for i in range(n-2, -1, -1):
            up = _resize_to(td[i+1], feats[i].shape[-2:])
            x = self.td_weights[i]([feats[i], up])
            td[i] = self.top_down_convs[i](x)
        outs = [None] * n
        outs[0] = self.bottom_up_convs[0](self.out_weights_first([feats[0], td[0]]))
        for i in range(1, n):
            down = F.max_pool2d(outs[i-1], 2, 2)
            x = self.out_weights[i-1]([feats[i], td[i], down])
            outs[i] = self.bottom_up_convs[i](x)
        return outs

class BiFPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int, num_layers: int = 3, norm: str = "bn", act: str = "silu"):
        super().__init__()
        self.num_layers = num_layers
        self.num_levels = len(in_channels_list)
        self.input_proj = nn.ModuleList([Conv1x1BnAct(c, out_channels, norm, act) for c in in_channels_list])
        self.layers = nn.ModuleList([BiFPNLayer(out_channels, self.num_levels, norm, act) for _ in range(num_layers)])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        x = [self.input_proj[i](inputs[i]) for i in range(self.num_levels)]
        for layer in self.layers:
            x = layer(x)
        return x
