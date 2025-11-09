"""from HRNet/HRNet-Semantic-Segmentation OCR implementation and from 'Object-Contextual Representations for Semantic Segmentation' (Yuan et al., ECCV 2020, arXiv:1909.11065)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGatherModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, probs):
        b, c, h, w = feats.shape
        k = probs.shape[1]
        feats = feats.view(b, c, -1)
        probs = probs.view(b, k, -1)
        probs = F.softmax(probs, dim=2)
        context = torch.bmm(feats, probs.transpose(1, 2))
        return context


class ObjectAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, scale=1):
        super().__init__()
        self.scale = scale
        self.pool = nn.Identity() if scale == 1 else nn.MaxPool2d(kernel_size=scale)
        self.f_pixel = nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False)
        self.f_object = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.g_object = nn.Conv1d(in_channels, value_channels, kernel_size=1, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(value_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        x = self.pool(x)
        b, c, h, w = x.shape
        query = self.f_pixel(x).view(b, -1, h * w).transpose(1, 2)
        key = self.f_object(proxy)
        sim_map = torch.bmm(query, key)
        sim_map = F.softmax(sim_map, dim=-1)
        value = self.g_object(proxy).transpose(1, 2)
        context = torch.bmm(sim_map, value).transpose(1, 2).contiguous().view(b, -1, h, w)
        context = self.out(context)
        if self.scale != 1:
            context = F.interpolate(context, size=(x.shape[2] * self.scale, x.shape[3] * self.scale), mode="bilinear", align_corners=False)
        return context

class OCRAttention(nn.Module):
    def __init__(self, in_channels, num_classes, key_channels=256, value_channels=256, dropout=0.05, scale=1):
        super().__init__()
        self.aux_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.gather = SpatialGatherModule()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, value_channels, scale=scale)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        proxy_logits = self.aux_head(x)
        context_vectors = self.gather(x, proxy_logits)
        context_map = self.object_context_block(x, context_vectors)
        out = self.fuse(torch.cat([x, context_map], dim=1))
        return out
