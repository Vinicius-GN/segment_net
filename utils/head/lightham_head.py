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


class LightHamBlock(nn.Module):
    """
    Large-kernel depthwise conv + pointwise + channel mixing MLP.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_fn: str = "batch_norm",
        dilation: int = 1,
    ):
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation
        self.dw = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.n1 = _make_norm(norm_fn, channels)
        self.act = nn.GELU()

        hidden = max(16, int(channels * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout, inplace=False),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.n2 = _make_norm(norm_fn, channels)

        self.gamma1 = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma2 = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor):
        y = self.dw(x)
        y = self.pw(y)
        y = self.n1(y)
        y = self.act(y)
        x = x + self.gamma1 * y
        y2 = self.mlp(x)
        y2 = self.n2(y2)
        return x + self.gamma2 * y2


class LightHamHead(nn.Module):
    """
    LightHam decode head with 3x3 fusion, N LightHamBlocks, and a classifier.
    Based on SegNeXt's convolutional attention idea and a lightweight 'Ham' style head. 
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

        self.num_blocks = int(self.config.get("head").get("lightham_num_blocks", 3))
        self.kernel_size = int(self.config.get("head").get("lightham_kernel", 15))
        self.mlp_ratio = float(self.config.get("head").get("lightham_mlp_ratio", 4.0))
        self.dw_dilation = int(self.config.get("head").get("lightham_dw_dilation", 1))
        self.fuse_kernel = int(self.config.get("head").get("fusion_kernel", 3))

        self._built = False
        self._in_ch = None

        self.fuse = None
        self.fuse_norm = None
        self.blocks = None
        self.dropout = nn.Dropout(self.dropout_prob, inplace=False)
        self.classifier = None
        self.act = nn.GELU()

    def _maybe_build(self, in_ch: int, device, dtype):
        if self._built and self._in_ch == in_ch:
            return
        pad = (self.fuse_kernel - 1) // 2
        self.fuse = nn.Conv2d(in_ch, in_ch, kernel_size=self.fuse_kernel, padding=pad, bias=False).to(device=device, dtype=dtype)
        self.fuse_norm = _make_norm(self.norm_fn, in_ch).to(device=device, dtype=dtype)

        blocks = []
        for _ in range(max(1, self.num_blocks)):
            blocks.append(
                LightHamBlock(
                    in_ch,
                    kernel_size=self.kernel_size,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout_prob,
                    norm_fn=self.norm_fn,
                    dilation=self.dw_dilation
                )
            )
        self.blocks = nn.Sequential(*blocks).to(device=device, dtype=dtype)

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

        z = self.fuse_norm(self.fuse(x_in))
        z = self.act(z)
        z = self.blocks(z)
        z = self.dropout(z)

        if self.opt_latency:
            z = self.classifier(z)
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
        else:
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
            z = self.classifier(z)
        return z
