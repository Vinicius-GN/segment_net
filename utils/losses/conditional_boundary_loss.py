import torch
import torch.nn as nn
import torch.nn.functional as F

"""From D. Wu, Z. Guo, A. Li, C. Yu, C. Gao and N. Sang, "Conditional Boundary Loss for Semantic Segmentation"""

class ConditionalBoundaryLoss(nn.Module):
    def __init__(self, kernel_size:int=5, alpha:float=0.1, beta:float=0.5, reduction:str='mean', class_weights=None, warmup_floor:float=0.05):
        super().__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.class_weights = class_weights
        self.warmup_floor = warmup_floor

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        k = self.kernel_size
        pad = k // 2
        device = logits.device

        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

        y_onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        avgpool = F.avg_pool2d(y_onehot, kernel_size=k, stride=1, padding=pad)
        boundary_mask = ((avgpool > 0.0) & (avgpool < 1.0)).any(dim=1)

        unfold = nn.Unfold(kernel_size=k, padding=pad)
        K = k * k
        center_idx = K // 2

        feats = logits
        feats_flat = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        targets_flat = targets.reshape(B, -1)
        boundary_flat = boundary_mask.reshape(B, -1)

        labels_win = unfold(targets.unsqueeze(1).float()).long()             # (B, K, H*W)
        preds_win  = unfold(preds.unsqueeze(1).float()).long()               # (B, K, H*W)
        feats_win  = unfold(feats).reshape(B, C, K, H * W).permute(0, 3, 2, 1)  # (B, H*W, K, C)

        mask_not_center = torch.ones(K, device=device, dtype=torch.bool)
        mask_not_center[center_idx] = False
        mask_not_center = mask_not_center.view(1, -1, 1).expand(B, -1, H * W)

        correct_win = (preds_win == labels_win) & mask_not_center
        anchor_labels = targets_flat.unsqueeze(1).expand(-1, K, -1)
        pos_win = correct_win & (labels_win == anchor_labels)
        neg_win = correct_win & (labels_win != anchor_labels)

        n_pos = pos_win.sum(dim=1)                      # (B, H*W)
        n_pairs = (pos_win | neg_win).sum(dim=1)        # (B, H*W)

        pos_mask_f = pos_win.permute(0, 2, 1).unsqueeze(-1).float()          # (B, H*W, K, 1)
        e_vec = (feats_win * pos_mask_f).sum(dim=2) / n_pos.clamp_min(1).unsqueeze(-1).float()  # (B, H*W, C)

        valid_a2c = boundary_flat & (n_pos > 0)
        if valid_a2c.any():
            f_i = feats_flat[valid_a2c]                      # (N, C)
            y_i = targets_flat[valid_a2c]                    # (N,)
            e_i_pair = e_vec[valid_a2c].detach()             # (N, C)
            e_i_ce   = e_vec[valid_a2c]                      # (N, C)
            l_pair_i = (f_i - e_i_pair).pow(2).sum(dim=-1)
            if self.class_weights is not None:
                w = self.class_weights.to(device)[y_i]
                l_pair_i = l_pair_i * w
            l_sce_i = F.cross_entropy(e_i_ce, y_i, weight=self.class_weights, reduction='none')
            l_a2c = (l_pair_i + self.alpha * l_sce_i).mean()
        else:
            l_a2c = torch.zeros((), device=device, dtype=logits.dtype)

        valid_a2pn = boundary_flat & (n_pairs > 0)
        if valid_a2pn.any():
            f_i = feats_flat[valid_a2pn]                    # (M, C)
            z_all = feats_win[valid_a2pn]                   # (M, K, C)
            pos_sel = pos_win.permute(0, 2, 1)[valid_a2pn]  # (M, K)
            neg_sel = neg_win.permute(0, 2, 1)[valid_a2pn]  # (M, K)
            mask_any = (pos_sel | neg_sel).float()          # (M, K)

            f_i_n = F.normalize(f_i, dim=-1, eps=1e-6).unsqueeze(1)     # (M,1,C)
            z_all_n = F.normalize(z_all.detach(), dim=-1, eps=1e-6)     # (M,K,C)
            sim = (f_i_n * z_all_n).sum(dim=-1)                         # (M,K)

            tgt = torch.zeros_like(sim)
            tgt[pos_sel] = 1.0

            mse_pairs = ((sim - tgt) ** 2 * mask_any).sum(dim=1) / mask_any.sum(dim=1).clamp_min(1.0)
            l_a2pn = mse_pairs.mean()
        else:
            l_a2pn = torch.zeros((), device=device, dtype=logits.dtype)

        l_cb = l_a2c + self.beta * l_a2pn

        frac_pos_on_boundary = (valid_a2c.sum().float() / boundary_flat.sum().clamp_min(1)).item()
        scale = 1.0 if frac_pos_on_boundary >= self.warmup_floor else (frac_pos_on_boundary / self.warmup_floor)
        l_cb = l_cb * float(scale)

        if self.reduction == 'mean':
            return l_cb
        elif self.reduction == 'sum':
            return l_cb * B
        else:
            return l_cb
