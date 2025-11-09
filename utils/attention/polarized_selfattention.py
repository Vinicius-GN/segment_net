"""from 'Polarized Self-Attention: Towards High-quality Pixel-wise Regression' (Liu et al., 2021) arXiv:2107.00782"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelPolarizedSelfAttention(nn.Module):
    def __init__(self, channel, reduction=2, eps=1e-6):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.q = nn.Conv2d(channel, 1, kernel_size=1, bias=True)
        self.v = nn.Conv2d(channel, hidden, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(hidden, channel, kernel_size=1, bias=True)
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).view(b, 1, -1)
        q = F.softmax(q, dim=-1)
        v = self.v(x).view(b, -1, h * w)
        z = torch.bmm(q, v.transpose(1, 2)).view(b, -1, 1, 1)
        z = self.proj(z)
        z = z.permute(0, 2, 3, 1).contiguous()
        z = F.layer_norm(z, (z.shape[-1],), eps=self.eps)
        z = z.permute(0, 3, 1, 2).contiguous()
        z = torch.sigmoid(z)
        return x * z


class SpatialPolarizedSelfAttention(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.q = nn.Conv2d(channel, hidden, kernel_size=1, bias=True)
        self.v = nn.Conv2d(channel, hidden, kernel_size=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x)
        q = F.adaptive_avg_pool2d(q, 1).view(b, -1, 1)
        q = F.softmax(q, dim=1)
        v = self.v(x).view(b, -1, h * w)
        z = torch.bmm(q.transpose(1, 2), v).view(b, 1, h, w)
        z = torch.sigmoid(z)
        return x * z


class ParallelPolarizedSelfAttention(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.ca = ChannelPolarizedSelfAttention(channel, reduction=reduction)
        self.sa = SpatialPolarizedSelfAttention(channel, reduction=reduction)

    def forward(self, x):
        xc = self.ca(x)
        xs = self.sa(x)
        return xc + xs


class SequentialPolarizedSelfAttention(nn.Module):
    def __init__(self, channel, reduction=2, order="cs"):
        super().__init__()
        self.ca = ChannelPolarizedSelfAttention(channel, reduction=reduction)
        self.sa = SpatialPolarizedSelfAttention(channel, reduction=reduction)
        self.order = order

    def forward(self, x):
        if self.order == "cs":
            return self.sa(self.ca(x))
        else:
            return self.ca(self.sa(x))
