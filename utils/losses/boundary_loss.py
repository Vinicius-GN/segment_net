import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, einsum
from typing import List

from scipy.ndimage import distance_transform_edt


def compute_sdf_per_class(gt_onehot):
    """
    Compute signed distance maps for each class mask in one-hot ground truth.
    gt_onehot: tensor [B, C, H, W] — one-hot encoded labels
    returns: sdf: [B, C, H, W]
    """
    gt_np = gt_onehot.cpu().numpy()
    sdf = torch.zeros_like(gt_onehot, dtype=torch.float32)

    for b in range(gt_np.shape[0]):
        for c in range(gt_np.shape[1]):
            posmask = gt_np[b, c].astype(bool)
            negmask = ~posmask
            posdist = distance_transform_edt(posmask)
            negdist = distance_transform_edt(negmask)
            sdf[b, c] = torch.from_numpy(negdist - posdist)

    return sdf

class MulticlassBoundaryLoss(nn.Module):
    def __init__(self, class_weights=None, reduction='mean', max_distance:int=20):
        super(MulticlassBoundaryLoss, self).__init__()
        
        self.class_weights = class_weights
        self.reduction = reduction
        self.max_distance = max_distance

    def forward(self, logits, targets):
        """
        logits: predicted probabilities [B, C, H, W] 
        targets: one-hot ground truth [B, H, W]
        """
        
        C = logits.shape[1]
        probs = logits.softmax(dim=1)
        
        # [B, H, W] -> [B, C, H, W]
        targets = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
                
        sdf = compute_sdf_per_class(targets).to(probs.device)
        
        # only positive
        # sdf = sdf.abs()
        #normaline [-1,1]
        # sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min())
        # sdf = 2*sdf -1 
        
        # log-based scaling
        # sdf = sdf.sign() * torch.log1p(sdf.abs()) / torch.log1p(self.max_distance)

        # clamp 
        sdf = sdf.clamp(-self.max_distance, self.max_distance) / self.max_distance
        
        per_class_loss = (probs * sdf).view(probs.shape[0], probs.shape[1], -1).mean(dim=2)

        # optionally weight classes
        if self.class_weights is not None:
            per_class_loss = per_class_loss * self.class_weights.view(1, -1)

        loss = per_class_loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 