
from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogCoshDiceLoss(nn.Module):
    def __init__(self, use_sigmoid: bool = False, reduction: str = 'mean'):
        super(LogCoshDiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_sigmoid:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = torch.softmax(inputs, dim=1)
        
        if targets.ndim == 3 and inputs.ndim == 4 and inputs.shape[1] > 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()

        elif targets.ndim == 3 and inputs.ndim == 4 and inputs.shape[1] == 1:
            targets = targets.unsqueeze(1).float()
    
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        log_cosh_dice = torch.log(torch.cosh(dice_score - 1.0))
        
        if self.reduction == 'mean':
            return log_cosh_dice.mean()
        elif self.reduction == 'sum':
            return log_cosh_dice.sum()
        else:
            return log_cosh_dice
