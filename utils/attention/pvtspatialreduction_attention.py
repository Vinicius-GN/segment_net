"""from 'Pyramid Vision Transformer' (Wang et al., 2021, arXiv:2102.12122) and 'PVTv2: Improved Baselines with Pyramid Vision Transformer' (Wang et al., 2021/2022, arXiv:2106.13797)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PVTSpatialReductionAttention"]


def _chunked_attn(q, k, v, scale, p_drop, chunk_tokens: int = 2048, training: bool = True):
    B, H, N, D = q.shape
    S = k.shape[2]
    outs = []
    for i in range(0, N, chunk_tokens):
        qc = q[:, :, i:i + chunk_tokens, :]
        attn = torch.matmul(qc, k.transpose(-2, -1)) * scale
        if training and p_drop > 0:
            attn = F.dropout(attn.softmax(dim=-1), p=p_drop, training=True)
        else:
            attn = attn.softmax(dim=-1)
        outs.append(torch.matmul(attn, v))
    return torch.cat(outs, dim=2)


class SRABlock(nn.Module):
    def __init__(self, in_channels, embed_dim=None, num_heads=8, sr_ratio=2, attn_drop=0.0, proj_drop=0.0, chunk_size=2048):
        super().__init__()
        embed_dim = in_channels if embed_dim is None else embed_dim
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=True)
        self.sr = nn.Conv2d(in_channels, in_channels, kernel_size=sr_ratio, stride=sr_ratio, bias=True) if sr_ratio > 1 else nn.Identity()
        self.norm = nn.LayerNorm(in_channels)
        self.kv = nn.Conv2d(in_channels, embed_dim * 2, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.chunk_size = chunk_size

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2).contiguous()
        xs = self.sr(x)
        hs, ws = xs.shape[2], xs.shape[3]
        xsn = xs.view(b, c, -1).permute(0, 2, 1).contiguous()
        xsn = self.norm(xsn).permute(0, 2, 1).contiguous().view(b, c, hs, ws)
        kv = self.kv(xsn)
        k, v = kv.chunk(2, dim=1)
        k = k.view(b, self.num_heads, self.head_dim, hs * ws).permute(0, 1, 3, 2).contiguous()
        v = v.view(b, self.num_heads, self.head_dim, hs * ws).permute(0, 1, 3, 2).contiguous()
        p_drop = self.attn_drop.p if self.training and self.attn_drop.p > 0 else 0.0
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=p_drop, is_causal=False)
        except Exception:
            out = _chunked_attn(q, k, v, self.scale, p_drop, self.chunk_size, self.training)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.embed_dim, h, w)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class LinearSRABlock(nn.Module):
    def __init__(self, in_channels, embed_dim=None, num_heads=8, pool_size=7, attn_drop=0.0, proj_drop=0.0, chunk_size=2048):
        super().__init__()
        embed_dim = in_channels if embed_dim is None else embed_dim
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=True)
        self.pool_size = pool_size
        self.norm = nn.LayerNorm(in_channels)
        self.kv = nn.Conv2d(in_channels, embed_dim * 2, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.chunk_size = chunk_size

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2).contiguous()
        xs = F.adaptive_avg_pool2d(x, output_size=(self.pool_size, self.pool_size))
        hs, ws = xs.shape[2], xs.shape[3]
        xsn = xs.view(b, c, -1).permute(0, 2, 1).contiguous()
        xsn = self.norm(xsn).permute(0, 2, 1).contiguous().view(b, c, hs, ws)
        kv = self.kv(xsn)
        k, v = kv.chunk(2, dim=1)
        k = k.view(b, self.num_heads, self.head_dim, hs * ws).permute(0, 1, 3, 2).contiguous()
        v = v.view(b, self.num_heads, self.head_dim, hs * ws).permute(0, 1, 3, 2).contiguous()
        p_drop = self.attn_drop.p if self.training and self.attn_drop.p > 0 else 0.0
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=p_drop, is_causal=False)
        except Exception:
            out = _chunked_attn(q, k, v, self.scale, p_drop, self.chunk_size, self.training)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.embed_dim, h, w)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class PVTSpatialReductionAttention(nn.Module):
    def __init__(self, channels, embed_dim=None, num_heads=8, mode="sra", sr_ratio=2, pool_size=7, attn_drop=0.0, proj_drop=0.0, chunk_size=2048):
        super().__init__()
        if mode == "linear":
            self.block = LinearSRABlock(channels, embed_dim, num_heads, pool_size, attn_drop, proj_drop, chunk_size)
        else:
            self.block = SRABlock(channels, embed_dim, num_heads, sr_ratio, attn_drop, proj_drop, chunk_size)

    def forward(self, x):
        return self.block(x)
