import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from scipy.ndimage import distance_transform_edt

# Active Boundary Loss (ABL) - Wang et al., AAAI'22
# https://doi.org/10.1609/aaai.v36i2.20139
# arXiv:2102.02696

def _pad_shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:

    b, c, h, w = x.shape
    pl, pr = max(dx, 0), max(-dx, 0)
    pt, pb = max(dy, 0), max(-dy, 0)
    y = F.pad(x, (pl, pr, pt, pb), mode="replicate")
    y = y[:, :, pb:pb + h, pr:pr + w]
    return y

def _neighbors8(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    offs = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    return tuple(_pad_shift(x, dx, dy) for dx, dy in offs)

def _neighbors2(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return (_pad_shift(x, 1, 0), _pad_shift(x, 0, 1))

def _kl_pq(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=1)

def _make_gt_boundary_map(targets: torch.Tensor) -> torch.Tensor:
    right = _pad_shift(targets.unsqueeze(1).float(), 1, 0).squeeze(1)
    down  = _pad_shift(targets.unsqueeze(1).float(), 0, 1).squeeze(1)
    br = (targets != right.long())
    bd = (targets != down.long())
    b = (br | bd).float()
    return b

def _distance_to_gtb(gt_boundary: torch.Tensor) -> torch.Tensor:
    gt_np = gt_boundary.detach().cpu().numpy()
    dist = []
    for b in range(gt_np.shape[0]):
        m = gt_np[b]
        d = distance_transform_edt(1.0 - m) 
        dist.append(torch.from_numpy(d))
    dist = torch.stack(dist, dim=0).unsqueeze(1).float() 
    return dist

def _topk_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    b, _, h, w = scores.shape
    k = max(1, int(ratio * h * w))
    sm = scores.reshape(b, -1)
    thresh = torch.topk(sm, k, dim=1).values[:, -1].unsqueeze(1)
    m = (sm >= thresh).float().reshape(b, 1, h, w)
    return m

class ActiveBoundaryLoss(nn.Module):
    def __init__(
        self,
        boundary_ratio: float = 0.01,
        theta: float = 20.0,
        label_smoothing_max: float = 0.0,
        use_2n_for_pb: bool = True,
        dilate_pb: int = 0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.boundary_ratio = float(boundary_ratio)
        self.theta = float(theta)
        self.label_smoothing_max = float(label_smoothing_max)
        self.use_2n_for_pb = bool(use_2n_for_pb)
        self.dilate_pb = int(dilate_pb)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        device = logits.device

        probs = logits.softmax(dim=1) 

        neigh_pb = _neighbors2(probs) if self.use_2n_for_pb else _neighbors8(probs)
        kl_pb = torch.stack([_kl_pq(probs, n.detach()) for n in neigh_pb], dim=1) 
        kl_pb = kl_pb.amax(dim=1, keepdim=True)  
        pb_mask = _topk_mask(kl_pb, self.boundary_ratio)  
        if self.dilate_pb > 0:
            k = 2 * self.dilate_pb + 1
            pb_mask = F.max_pool2d(pb_mask, kernel_size=k, stride=1, padding=self.dilate_pb)

        gtb = _make_gt_boundary_map(targets)             
        dist = _distance_to_gtb(gtb).to(device)           
        w = (dist.clamp_max(self.theta) / max(self.theta, 1e-6))  

      
        offs = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
        neigh_dist = torch.stack([_pad_shift(dist, dx, dy) for dx, dy in offs], dim=1)  
        dgi = neigh_dist.argmin(dim=1).squeeze(1)  

        neigh8 = _neighbors8(probs)
        kl_logits = torch.stack([_kl_pq(probs, n.detach()) for n in neigh8], dim=1)  
        logp = F.log_softmax(kl_logits, dim=1)  

        K = 8
        oh = F.one_hot(dgi.long(), num_classes=K).permute(0, 3, 1, 2).float() 

        if self.label_smoothing_max and self.label_smoothing_max > 0:
            eps = float(self.label_smoothing_max)
            q = oh * (1.0 - eps) + (1.0 - oh) * (eps / (K - 1))
        else:
            q = oh

        ce_map = -(q * logp).sum(dim=1, keepdim=True)  # (B,1,H,W)

        num = (ce_map * w * pb_mask).reshape(B, -1).sum(dim=1)
        den = pb_mask.reshape(B, -1).sum(dim=1).clamp_min(1.0)
        loss_b = num / den  # (B,)

        if self.reduction == "mean":
            return loss_b.mean()
        elif self.reduction == "sum":
            return loss_b.sum()
        else:
            return loss_b
