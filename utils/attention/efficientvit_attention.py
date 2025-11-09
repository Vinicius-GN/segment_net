"""from 'EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction' (Cai et al., ICCV 2023, arXiv:2205.14756)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientViTMSA(nn.Module):
    def __init__(self, channels, head_dim=32, head_kernel_sizes=(3, 5), eps=1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.head_kernel_sizes = tuple(head_kernel_sizes)
        self.num_heads = len(self.head_kernel_sizes)
        self.inner_dim = self.head_dim * self.num_heads
        self.qkv = nn.Conv2d(channels, self.inner_dim * 3, kernel_size=1, bias=True)
        self.dw_q = nn.ModuleList([nn.Conv2d(self.head_dim, self.head_dim, k, padding=k // 2, groups=self.head_dim, bias=True) for k in self.head_kernel_sizes])
        self.dw_k = nn.ModuleList([nn.Conv2d(self.head_dim, self.head_dim, k, padding=k // 2, groups=self.head_dim, bias=True) for k in self.head_kernel_sizes])
        self.dw_v = nn.ModuleList([nn.Conv2d(self.head_dim, self.head_dim, k, padding=k // 2, groups=self.head_dim, bias=True) for k in self.head_kernel_sizes])
        self.gq = nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=1, groups=self.num_heads, bias=True)
        self.gk = nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=1, groups=self.num_heads, bias=True)
        self.gv = nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=1, groups=self.num_heads, bias=True)
        self.proj = nn.Conv2d(self.inner_dim, channels, kernel_size=1, bias=True)
        self.eps = eps

    def _aggregate(self, t, dw_list, gconv):
        chunks = t.chunk(self.num_heads, dim=1)
        chunks = [dw_list[i](chunks[i]) for i in range(self.num_heads)]
        t = torch.cat(chunks, dim=1)
        t = gconv(t)
        return t

    def _linear_attn(self, q, k, v):
        b, d_all, h, w = q.shape
        hds = self.head_dim
        nh = self.num_heads
        n = h * w
        q = q.view(b, nh, hds, h, w).permute(0, 1, 3, 4, 2).reshape(b, nh, n, hds)
        k = k.view(b, nh, hds, h, w).permute(0, 1, 3, 4, 2).reshape(b, nh, n, hds)
        v = v.view(b, nh, hds, h, w).permute(0, 1, 3, 4, 2).reshape(b, nh, n, hds)
        q = F.relu(q, inplace=False)
        k = F.relu(k, inplace=False)
        kv = torch.matmul(k.transpose(-2, -1), v)
        z = k.sum(dim=-2).unsqueeze(-1)
        num = torch.matmul(q, kv)
        den = torch.matmul(q, z).clamp(min=self.eps)
        out = num / den
        out = out.reshape(b, nh, h, w, hds).permute(0, 1, 4, 2, 3).contiguous().view(b, d_all, h, w)
        return out

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.inner_dim, dim=1)
        q = self._aggregate(q, self.dw_q, self.gq)
        k = self._aggregate(k, self.dw_k, self.gk)
        v = self._aggregate(v, self.dw_v, self.gv)
        y = self._linear_attn(q, k, v)
        y = self.proj(y)
        return y
