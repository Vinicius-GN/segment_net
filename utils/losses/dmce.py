import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

"""Distance Map Loss Penalty Term for Semantic Segmentation", MIDL 2019 / arXiv:1908.03679"""

def _inverse_distance_to_boundary(onehot):
    B, C, H, W = onehot.shape
    inv = torch.zeros((B, 1, H, W), device=onehot.device, dtype=onehot.dtype)
    t = onehot.detach().cpu().numpy().astype(bool)
    for b in range(B):
        dmin = np.zeros((H, W), dtype=np.float32)
        for c in range(C):
            fg = t[b, c]
            if fg.any() and (~fg).any():
                d_fg = distance_transform_edt(fg)
                d_bg = distance_transform_edt(~fg)
                d = np.minimum(d_fg, d_bg)
            else:
                d = np.zeros((H, W), dtype=np.float32)
            dmin = np.maximum(dmin, d)
        inv_b = 1.0 / (1.0 + dmin)
        inv[b, 0] = torch.from_numpy(inv_b)
    return inv

class DistanceMapCELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        phi = _inverse_distance_to_boundary(onehot)
        w = 1.0 + phi
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.0)
        loss = w.squeeze(1) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
