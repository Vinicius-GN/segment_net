import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKLoss(nn.Module):
    def __init__(self, ignore_index=255, k_frac=0.1, gamma=0.0):
        super(TopKLoss, self).__init__()
        self.ignore_index = ignore_index
        self.k_frac = k_frac   
        self.gamma = gamma     

    def forward(self, input, target):
        mask = target != self.ignore_index
        input = input.permute(0, 2, 3, 1)[mask]  # N*H*W, C
        target = target[mask]  # N*H*W

        ce = F.cross_entropy(input, target, reduction='none')

        if self.gamma > 0:
            pt = torch.gather(F.softmax(input, dim=1), 1, target.unsqueeze(1)).squeeze(1)
            ce = (1 - pt).pow(self.gamma) * ce

        # Top-K selection
        k = max(1, int(len(ce) * self.k_frac))
        topk_loss, _ = torch.topk(ce, k, largest=True)
        return topk_loss.mean()