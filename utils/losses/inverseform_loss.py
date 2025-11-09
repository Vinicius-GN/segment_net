import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

#From "https://arxiv.org/pdf/2104.02745" - InverseForm: A Loss Function for Structured Boundary-Aware Segmentation

def _binary_boundary_from_labels(y: torch.Tensor) -> torch.Tensor:
    y = y.long()
    y1 = y.unsqueeze(1)
    r = F.pad(y1, (1, 0, 0, 0), mode="replicate").squeeze(1)[:, :, :y.shape[2]]
    d = F.pad(y1, (0, 0, 1, 0), mode="replicate").squeeze(1)[:, :y.shape[1], :]
    br = (y != r).float()
    bd = (y != d).float()
    return ((br + bd) > 0).float().unsqueeze(1)

def _binary_boundary_from_logits(logits: torch.Tensor) -> torch.Tensor:
    y = logits.argmax(dim=1)
    return _binary_boundary_from_labels(y)

def _tile(x: torch.Tensor, k: int, s: Optional[int] = None) -> torch.Tensor:
    if s is None:
        s = k
    unfold = nn.Unfold(kernel_size=k, stride=s, padding=0)
    p = unfold(x)
    b, _, h, w = x.shape
    n = p.shape[-1]
    return p.transpose(1, 2).reshape(b * n, 1, k, k), n, h, w

def _prep_pair_tiles(bpred: torch.Tensor, bgt: torch.Tensor, tile: int, stride: Optional[int], reduce: int) -> torch.Tensor:
    if reduce > 1:
        bpred = F.avg_pool2d(bpred, kernel_size=reduce, stride=reduce)
        bgt = F.avg_pool2d(bgt, kernel_size=reduce, stride=reduce)
    tp, n, _, _ = _tile(bpred, tile, stride)
    tg, _, _, _ = _tile(bgt, tile, stride)
    x = torch.cat([tp, tg], dim=1)
    x = x.flatten(1)
    return x, n

class InverseTransformNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

def _affine_from_vec(v: torch.Tensor) -> torch.Tensor:
    a, b, tx, c, d, ty = v.unbind(-1)
    m = torch.stack([a, b, tx, c, d, ty, torch.zeros_like(a), torch.zeros_like(a), torch.ones_like(a)], dim=-1)
    return m.view(-1, 3, 3)

def _homography_from_vec(v: torch.Tensor) -> torch.Tensor:
    h11, h12, h13, h21, h22, h23, h31, h32 = v.unbind(-1)
    m = torch.stack([h11, h12, h13, h21, h22, h23, h31, h32, torch.ones_like(h11)], dim=-1)
    return m.view(-1, 3, 3)

def _euclidean_distance_to_I(mat: torch.Tensor) -> torch.Tensor:
    i = torch.eye(3, device=mat.device, dtype=mat.dtype).unsqueeze(0).expand_as(mat)
    return (mat - i).pow(2).sum(dim=(1, 2)).sqrt()

def _geodesic_distance_proj_so3(h: torch.Tensor, lam: float = 0.1, eps: float = 1e-9) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(h, full_matrices=False)
    uv = u @ vh
    det = torch.det(uv).unsqueeze(-1).unsqueeze(-1)
    diag = torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1).squeeze(-3)
    p = u @ diag @ vh
    tr = p.diagonal(dim1=-2, dim2=-1).sum(-1)
    c = (tr - 1) * 0.5
    c = c.clamp(-1 + eps, 1 - eps)
    ang = torch.arccos(c)
    rpi = h - p
    res = ang + lam * (rpi.transpose(1, 2) @ rpi).diagonal(dim1=-2, dim2=-1).sum(-1)
    return res

class InverseFormLoss(nn.Module):
    def __init__(self, mode: str = "euclidean_affine", tile: int = 64, stride: Optional[int] = None, reduce: int = 4, lam_geo: float = 0.1, itn: Optional[nn.Module] = None, itn_weights: Optional[str] = None, freeze_itn: bool = True, reduction: str = "mean"):
        super().__init__()
        self.mode = mode
        self.tile = tile
        self.stride = stride
        self.reduce = reduce
        self.lam_geo = lam_geo
        self.reduction = reduction
        self.itn = itn
        self.itn_weights = itn_weights
        self.freeze_itn = freeze_itn
        self._built = False

    def _build(self, sample_shape, device):
        b, c, h, w = sample_shape
        in_dim = 2 * self.tile * self.tile
        out_dim = 6 if self.mode.startswith("euclidean") else 8
        if self.itn is None:
            self.itn = InverseTransformNet(in_dim=in_dim, hidden=512, out_dim=out_dim)
        if self.itn_weights is not None:
            sd = torch.load(self.itn_weights, map_location="cpu")
            self.itn.load_state_dict(sd, strict=False)
        self.itn = self.itn.to(device)
        if self.freeze_itn:
            for p in self.itn.parameters():
                p.requires_grad = False
            self.itn.eval()
        self._built = True

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        device = logits.device
        bpred = _binary_boundary_from_logits(logits)
        bgt = _binary_boundary_from_labels(targets)
        if not self._built:
            self._build(bpred.shape, device)
        else:
            self.itn = self.itn.to(device)
        x, ntiles = _prep_pair_tiles(bpred, bgt, self.tile, self.stride, self.reduce)
        x = x.to(device)
        mask = (x.sum(dim=1) > 0).float()
        v = self.itn(x)
        if self.mode.startswith("euclidean"):
            m = _affine_from_vec(v)
            d = _euclidean_distance_to_I(m)
        else:
            h = _homography_from_vec(v)
            d = _geodesic_distance_proj_so3(h, lam=self.lam_geo)
        d = d * mask
        d = d.view(-1, ntiles).sum(dim=1) / mask.view(-1, ntiles).sum(dim=1).clamp_min(1.0)
        if self.reduction == "mean":
            return d.mean()
        elif self.reduction == "sum":
            return d.sum()
        else:
            return d
